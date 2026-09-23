# Databricks notebook source
# MAGIC %md
# MAGIC # Online Retail II — Exploratory Data Analysis
# MAGIC
# MAGIC UK giftware retailer, transaction line items, Dec 2009 – Dec 2011.  
# MAGIC Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) / [Kaggle mirror](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).
# MAGIC
# MAGIC **Questions this notebook answers**
# MAGIC 1. What is the grain and quality of the dataset?
# MAGIC 2. How does revenue change over time?
# MAGIC 3. Which products drive sales?
# MAGIC 4. Which customers drive sales?
# MAGIC 5. What can we learn about cancellations/returns and geography?
# MAGIC
# MAGIC Run top to bottom. Numbers in **[brackets]** are filled after you run the cells that compute them — do not treat this first pass of the markdown as final results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install openpyxl -q

# COMMAND ----------

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Databricks provides display(); this fallback is only for local smoke-tests
try:
    display  # noqa: F821
except NameError:
    def display(obj, **_):
        if hasattr(obj, "limit"):
            print(obj.limit(50).toPandas().to_string(index=False))
        elif hasattr(obj, "to_string"):
            print(obj.head(50).to_string(index=False))
        else:
            print(obj)


def money(x):
    if x is None:
        return None
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"£{x/1_000_000:,.2f}M"
    if abs(x) >= 1_000:
        return f"£{x:,.0f}"
    return f"£{x:,.2f}"


# COMMAND ----------

# MAGIC %md
# MAGIC Load the two Excel sheets, keep a `source_sheet` marker, convert to a Spark DataFrame. Column names are taken from the file — not assumed.

# COMMAND ----------

UCI_ZIP = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SHEETS = ["Year 2009-2010", "Year 2010-2011"]

# Databricks Volume if you uploaded the file; otherwise download from UCI
CANDIDATE_PATHS = [
    Path("/Volumes/workspace/default/online_retail_II.xlsx"),
    Path("data/online_retail_II.xlsx"),
]


def load_excel() -> pd.DataFrame:
    xlsx = next((p for p in CANDIDATE_PATHS if p.exists()), None)
    if xlsx is None:
        print("Downloading UCI zip…")
        with urlopen(UCI_ZIP, timeout=120) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        xlsx_name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        xlsx = zf.read(xlsx_name)
        reader = io.BytesIO(xlsx)
    else:
        print(f"Reading {xlsx}")
        reader = xlsx

    frames = []
    for sheet in SHEETS:
        part = pd.read_excel(
            reader if not isinstance(reader, Path) else reader,
            sheet_name=sheet,
            dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
        )
        if isinstance(reader, io.BytesIO):
            reader.seek(0)
        part["source_sheet"] = sheet
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


pdf = load_excel()
print("pandas rows:", f"{len(pdf):,}")
print("source columns:", list(pdf.columns))

raw = spark.createDataFrame(pdf)
raw.createOrReplaceTempView("retail_raw")
print(f"Spark rows: {raw.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Executive summary
# MAGIC
# MAGIC _Fill these after running Sections 2–4. The `key_facts` table at the end of Section 4 is the source of truth._
# MAGIC
# MAGIC - **Grain:** each row is one invoice line item (not one invoice, not one customer).
# MAGIC - **Coverage:** [n_rows] rows, [n_invoices] invoices, [n_customers] identified customers, [n_countries] countries, [date_min] → [date_max].
# MAGIC - **Revenue (working definition — confirmed after DQ):** [gross] gross, [returns] returns, [net] net. See caveats if this includes postage / guest sales / the incomplete last month.
# MAGIC - **Time:** [one sentence on seasonality and like-for-like growth].
# MAGIC - **Customers / products:** [concentration one-liner].
# MAGIC - **Returns / geography:** [one-liner].
# MAGIC - **Most trusted finding / least certain finding:** [fill after Section 5].

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data overview
# MAGIC
# MAGIC Confirm names, types, grain, and entity counts **before** deriving `line_revenue`.

# COMMAND ----------

print("Schema (as Spark inferred it from the Excel file):")
raw.printSchema()

overview = spark.sql(
    """
    SELECT
      COUNT(*)                                              AS n_rows,
      COUNT(DISTINCT Invoice)                               AS n_invoices,
      COUNT(DISTINCT StockCode)                             AS n_stock_codes,
      COUNT(DISTINCT `Customer ID`)                         AS n_identified_customers,
      COUNT(DISTINCT Country)                               AS n_countries,
      MIN(InvoiceDate)                                      AS date_min,
      MAX(InvoiceDate)                                      AS date_max,
      datediff(MAX(InvoiceDate), MIN(InvoiceDate))          AS span_days
    FROM retail_raw
    """
)
display(overview)

display(
    raw.groupBy("source_sheet")
    .agg(
        F.count("*").alias("rows"),
        F.countDistinct("Invoice").alias("invoices"),
        F.min("InvoiceDate").alias("first_ts"),
        F.max("InvoiceDate").alias("last_ts"),
    )
    .orderBy("first_ts")
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Grain check.** If many rows share the same `(Invoice, StockCode)`, the grain is *line item* (a SKU can appear more than once on an invoice), not unique SKU-per-invoice.

# COMMAND ----------

grain = raw.groupBy("Invoice", "StockCode").count()
display(
    grain.agg(
        F.count("*").alias("invoice_sku_pairs"),
        F.sum(F.when(F.col("count") > 1, 1).otherwise(0)).alias("pairs_with_repeats"),
        F.max("count").alias("max_repeats_of_one_pair"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation — data overview (fill after the tables above)**
# MAGIC
# MAGIC - Columns present: [list from printSchema]. Price column name is **[Price / UnitPrice]** — only now do we create `line_revenue`.
# MAGIC - Apparent grain: [line item / something else].
# MAGIC - The two sheets [do / do not] overlap in calendar time: [dates]. That overlap is the first DQ risk.

# COMMAND ----------

# Column name confirmed from the file above. Do not create this earlier.
PRICE_COL = "Price" if "Price" in raw.columns else "UnitPrice"
assert PRICE_COL in raw.columns, f"No price column in {raw.columns}"

raw = raw.withColumn("line_revenue", F.col("Quantity") * F.col(PRICE_COL))
raw.createOrReplaceTempView("retail_raw")
print(f"Using price column: {PRICE_COL}")
display(raw.limit(8))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data quality checks
# MAGIC
# MAGIC Investigate **before** dropping or filtering anything. Each subsection ends with a decision, not a silent cleanup.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Nulls

# COMMAND ----------

nulls = spark.sql(
    f"""
    SELECT
      COUNT(*)                                            AS n_rows,
      SUM(CAST(Invoice IS NULL AS INT))                   AS invoice_null,
      SUM(CAST(StockCode IS NULL AS INT))                 AS stockcode_null,
      SUM(CAST(Description IS NULL AS INT))               AS description_null,
      SUM(CAST(Quantity IS NULL AS INT))                  AS quantity_null,
      SUM(CAST(InvoiceDate IS NULL AS INT))               AS invoicedate_null,
      SUM(CAST(`{PRICE_COL}` IS NULL AS INT))             AS price_null,
      SUM(CAST(`Customer ID` IS NULL AS INT))             AS customer_id_null,
      SUM(CAST(Country IS NULL AS INT))                   AS country_null,
      ROUND(AVG(CAST(`Customer ID` IS NULL AS INT)), 4)   AS customer_id_null_rate,
      ROUND(AVG(CAST(Description IS NULL AS INT)), 4)     AS description_null_rate
    FROM retail_raw
    """
)
display(nulls)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Duplicates — within a sheet, and across the two sheets

# COMMAND ----------

KEY = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", PRICE_COL, "Customer ID", "Country"]

exact_dup_extra = raw.count() - raw.dropDuplicates(KEY + ["source_sheet"]).count()
exact_dup_ignoring_sheet = raw.count() - raw.dropDuplicates(KEY).count()
print(f"Surplus exact copies including source_sheet: {exact_dup_extra:,}")
print(f"Surplus exact copies IGNORING source_sheet:  {exact_dup_ignoring_sheet:,}")
print("If the second number is much larger, the two sheets share the same rows.")

s1 = raw.filter((F.col("source_sheet") == "Year 2009-2010") & (F.col("InvoiceDate") >= "2010-12-01"))
s2 = raw.filter((F.col("source_sheet") == "Year 2010-2011") & (F.col("InvoiceDate") < "2010-12-10"))
print(f"Sheet 1 rows dated ≥ 2010-12-01: {s1.count():,}  invoices {s1.select('Invoice').distinct().count():,}  value {money(s1.agg(F.sum('line_revenue')).first()[0])}")
print(f"Sheet 2 rows dated 1–9 Dec 2010: {s2.count():,}  invoices {s2.select('Invoice').distinct().count():,}  value {money(s2.agg(F.sum('line_revenue')).first()[0])}")

display(raw.filter(F.col("Invoice") == s1.select("Invoice").first()[0]).orderBy("StockCode").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC **Decision on the sheet overlap.** If the two windows match row-for-row, a naïve union double-counts that week. For the rest of the notebook we drop the *sheet-1 copy* of 1–9 Dec 2010 only. We do **not** drop same-invoice repeated lines yet — those may be real extra scans.

# COMMAND ----------

df = raw.filter(~((F.col("source_sheet") == "Year 2009-2010") & (F.col("InvoiceDate") >= "2010-12-01")))
print(f"Rows after dropping the sheet-1 overlap copy: {raw.count():,} → {df.count():,}")
df.createOrReplaceTempView("retail")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Negative quantities, invoice prefixes, zero / negative prices
# MAGIC
# MAGIC Do **not** drop negative quantities yet. Invoices starting with `C` are the usual cancellation marker on this file; we check whether that is the whole story.

# COMMAND ----------

df = df.withColumn(
    "invoice_prefix",
    F.coalesce(F.regexp_extract("Invoice", r"^([A-Za-z]+)", 1), F.lit("(numeric)")),
)
df.createOrReplaceTempView("retail")

display(
    spark.sql(
        f"""
        SELECT
          invoice_prefix,
          COUNT(*)                                              AS rows,
          COUNT(DISTINCT Invoice)                               AS invoices,
          SUM(CAST(Quantity < 0 AS INT))                        AS neg_qty_rows,
          SUM(CAST(`{PRICE_COL}` < 0 AS INT))                   AS neg_price_rows,
          SUM(CAST(`{PRICE_COL}` = 0 AS INT))                   AS zero_price_rows,
          SUM(line_revenue)                                     AS net_line_revenue
        FROM retail
        GROUP BY invoice_prefix
        ORDER BY rows DESC
        """
    )
)

neg_not_c = df.filter((F.col("Quantity") < 0) & (F.col("invoice_prefix") == "(numeric)"))
print(f"Negative qty on non-C invoices: {neg_not_c.count():,}")
print(
    "  share price==0:",
    f"{neg_not_c.filter(F.col(PRICE_COL) == 0).count() / max(neg_not_c.count(), 1):.1%}",
    "  share missing customer:",
    f"{neg_not_c.filter(F.col('Customer ID').isNull()).count() / max(neg_not_c.count(), 1):.1%}",
)

print("Negative-price rows (all of them):")
display(df.filter(F.col(PRICE_COL) < 0).select("Invoice", "StockCode", "Description", "Quantity", PRICE_COL, "Customer ID", "InvoiceDate"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Missing Customer ID — sales vs warehouse notes

# COMMAND ----------

guest = df.filter(F.col("Customer ID").isNull())
guest_inv = guest.groupBy("Invoice").agg(
    F.sum("line_revenue").alias("value"),
    F.min(F.when(F.col(PRICE_COL) == 0, True).otherwise(False)).alias("any_zero"),
    (F.sum(F.when(F.col(PRICE_COL) == 0, 1).otherwise(0)) == F.count("*")).alias("all_zero_price"),
)
print(f"Rows with no Customer ID: {guest.count():,}  ({guest.count()/df.count():.1%} of rows)")
print(f"Invoices with no Customer ID: {guest_inv.count():,}")
print(f"  entirely £0 invoices: {guest_inv.filter('all_zero_price').count():,}")
print(f"  invoices with some positive price: {guest_inv.filter('NOT all_zero_price').count():,}")

display(
    guest.filter(F.col(PRICE_COL) == 0)
    .groupBy("Description")
    .count()
    .orderBy(F.desc("count"))
    .limit(12)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Extreme quantities and prices, and non-product stock codes

# COMMAND ----------

display(df.select("Quantity", PRICE_COL, "line_revenue").summary())

display(
    df.orderBy(F.desc(F.abs("Quantity")))
    .select("Invoice", "StockCode", "Description", "Quantity", PRICE_COL, "InvoiceDate", "Customer ID", "Country")
    .limit(12)
)

alpha = (
    df.filter(F.col("StockCode").rlike("^[A-Za-z]"))
    .groupBy("StockCode")
    .agg(
        F.count("*").alias("rows"),
        F.first("Description", ignorenulls=True).alias("example_description"),
        F.sum("line_revenue").alias("net_revenue"),
    )
    .orderBy(F.desc("rows"))
)
display(alpha.limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Time coverage — last month and operating hours

# COMMAND ----------

inv_ts = df.groupBy("Invoice").agg(F.min("InvoiceDate").alias("ts"))
inv_ts = inv_ts.withColumn("dow", F.date_format("ts", "E")).withColumn("hour", F.hour("ts"))

display(inv_ts.groupBy("dow").count().orderBy("dow"))
display(inv_ts.groupBy("hour").count().orderBy("hour"))

last = df.filter(F.col("InvoiceDate") >= "2011-12-01")
print(
    f"Last timestamp: {df.agg(F.max('InvoiceDate')).first()[0]}  |  "
    f"distinct days in Dec 2011: {last.select(F.dayofmonth('InvoiceDate')).distinct().count()}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC **DQ interpretation (fill after the tables)**
# MAGIC
# MAGIC | Check | What I saw | Decision for the rest of the notebook |
# MAGIC |---|---|---|
# MAGIC | Sheet overlap | [ ] | Drop sheet-1 copy of 1–9 Dec 2010 if they match; keep otherwise |
# MAGIC | Exact duplicate lines | [ ] | Keep unless value share is material — they may be real repeats |
# MAGIC | `C` invoices vs other negative qty | [ ] | Treat `C` as credits; inspect non-`C` negatives before calling them returns |
# MAGIC | Negative / zero price | [ ] | Exclude accounting adjustments and £0 notes from merchandise revenue |
# MAGIC | Missing Customer ID | [ ] | Keep in revenue; drop from customer-level metrics |
# MAGIC | Extreme qty | [ ] | Check for a matching `C` invoice the same day before ranking products |
# MAGIC | Alpha stock codes | [ ] | Decide whether POST / DOT / M / AMAZONFEE belong in merchandise sales |
# MAGIC | Dec 2011 / Saturday | [ ] | Do not compare a partial last month to a full month |
# MAGIC
# MAGIC **Working views used below** (not a pipeline — just named filters so later cells stay readable):

# COMMAND ----------

# Flags justified by the tables above. Tune NON_PRODUCT after looking at §3.5.
NON_PRODUCT = r"^(POST|DOT|M|C2|D|S|BANK CHARGES|ADJUST\d*|AMAZONFEE|CRUK|TEST\d*|gift_\w+|PADS|B)$"

df = (
    df.withColumn("is_cancel", F.col("invoice_prefix") == "C")
    .withColumn("is_adjust", F.col("invoice_prefix") == "A")
    .withColumn("is_anonymous", F.col("Customer ID").isNull())
    .withColumn("is_non_product", F.col("StockCode").rlike(NON_PRODUCT))
    .withColumn("is_zero_price", F.col(PRICE_COL) == 0)
)

# Merchandise trade: a real unit price, not an accounting prefix, not a fee/postage/manual code.
# Negative quantities on C invoices stay in — they are the returns.
trade = df.filter(~F.col("is_adjust") & ~F.col("is_zero_price") & ~F.col("is_non_product"))
sales = trade.filter(~F.col("is_cancel") & (F.col("Quantity") > 0))
returns = trade.filter(F.col("is_cancel"))

sales.createOrReplaceTempView("sales")
returns.createOrReplaceTempView("returns")
trade.createOrReplaceTempView("trade")

waterfall = spark.createDataFrame(
    [
        ("1 raw union of both sheets", raw.count(), float(raw.agg(F.sum("line_revenue")).first()[0] or 0)),
        ("2 after sheet-overlap drop", df.count(), float(df.agg(F.sum("line_revenue")).first()[0] or 0)),
        ("3 merchandise trade (sales + C-invoice credits)", trade.count(), float(trade.agg(F.sum("line_revenue")).first()[0] or 0)),
        ("3a   of which sales (qty>0, not C)", sales.count(), float(sales.agg(F.sum("line_revenue")).first()[0] or 0)),
        ("3b   of which C-invoice credits", returns.count(), float(returns.agg(F.sum("line_revenue")).first()[0] or 0)),
    ],
    ["step", "rows", "net_line_revenue"],
)
display(waterfall)

print(
    "Return rate on this definition:",
    f"{-(returns.agg(F.sum('line_revenue')).first()[0] or 0) / (sales.agg(F.sum('line_revenue')).first()[0] or 1):.1%} of gross merchandise",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Exploratory analysis
# MAGIC
# MAGIC Five views only. Charts use Databricks `display()` — in the result, switch to **Line** / **Bar** where noted.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 How does revenue change over time?
# MAGIC
# MAGIC Dec 2011 looks short in §3.6, so like-for-like windows are **Dec–Nov vs Dec–Nov**, not calendar years.

# COMMAND ----------

monthly = spark.sql(
    """
    SELECT
      date_trunc('month', InvoiceDate)                      AS month,
      SUM(CASE WHEN Quantity > 0 THEN line_revenue ELSE 0 END) AS gross,
      SUM(CASE WHEN Quantity < 0 THEN line_revenue ELSE 0 END) AS returns,
      SUM(line_revenue)                                     AS net,
      COUNT(DISTINCT Invoice)                               AS invoices,
      COUNT(DISTINCT `Customer ID`)                         AS identified_customers
    FROM trade
    GROUP BY 1
    ORDER BY 1
    """
)
display(monthly)  # Visualisation: Line, x=month, y=gross and net

# COMMAND ----------

yoy = spark.sql(
    """
    WITH tagged AS (
      SELECT *,
        CASE
          WHEN InvoiceDate >= '2009-12-01' AND InvoiceDate < '2010-12-01' THEN 'Dec09–Nov10'
          WHEN InvoiceDate >= '2010-12-01' AND InvoiceDate < '2011-12-01' THEN 'Dec10–Nov11'
        END AS window
      FROM trade
    )
    SELECT
      window,
      SUM(CASE WHEN Quantity > 0 AND NOT is_cancel THEN line_revenue END) AS gross,
      SUM(CASE WHEN is_cancel THEN line_revenue END)                      AS returns,
      SUM(line_revenue)                                                   AS net,
      COUNT(DISTINCT CASE WHEN Quantity > 0 AND NOT is_cancel THEN Invoice END) AS sale_invoices,
      COUNT(DISTINCT `Customer ID`)                                       AS identified_customers,
      SUM(CASE WHEN Quantity > 0 AND NOT is_cancel THEN Quantity END)     AS units
    FROM tagged
    WHERE window IS NOT NULL
    GROUP BY window
    ORDER BY window
    """
)
display(yoy)

# COMMAND ----------

# MAGIC %md
# MAGIC **Time interpretation (fill)**
# MAGIC
# MAGIC - Seasonality: [e.g. Q4 share, Nov vs Feb].
# MAGIC - Like-for-like Dec–Nov: gross [ ], net [ ], invoices [ ], units [ ].
# MAGIC - Dec 2011: [partial month — do not call it a collapse].
# MAGIC - Confidence: [ ]. Caveat: [guest sales / postage exclusion / only two seasons].

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Which products drive sales?
# MAGIC
# MAGIC Rank on **net** merchandise value so a same-day cancel cannot become a top product.

# COMMAND ----------

products = spark.sql(
    """
    SELECT
      StockCode,
      FIRST(Description, TRUE) AS description,
      SUM(line_revenue)        AS net_revenue,
      SUM(Quantity)            AS net_units,
      COUNT(DISTINCT Invoice)  AS invoices
    FROM trade
    GROUP BY StockCode
    """
)
products = products.filter("net_revenue > 0").orderBy(F.desc("net_revenue"))
w = Window.orderBy(F.desc("net_revenue"))
products = products.withColumn("rank", F.row_number().over(w)).withColumn(
    "cum_share", F.sum("net_revenue").over(w) / F.sum("net_revenue").over(Window.partitionBy())
)
display(products.limit(15))  # Visualisation: Bar, x=description, y=net_revenue

print("SKUs with positive net:", products.count())
print("Share of net from top 10:", products.filter("rank <= 10").agg(F.max("cum_share")).first()[0])
print("SKUs needed for ~50% of net:", products.filter("cum_share <= 0.50").count())
print("SKUs needed for ~80% of net:", products.filter("cum_share <= 0.80").count())

# COMMAND ----------

# MAGIC %md
# MAGIC **Product interpretation (fill)**
# MAGIC
# MAGIC - Top product and its share: [ ].
# MAGIC - How long is the tail (SKUs for 50% / 80%): [ ].
# MAGIC - Did any “top product” disappear once we netted C-invoices? [ ].
# MAGIC - Confidence: [ ]. Caveat: descriptions are not a clean product master; `M` / postage were excluded by `NON_PRODUCT`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Which customers drive sales?
# MAGIC
# MAGIC Identified customers only. Anonymous lines stay in the revenue totals in §4.1 and are excluded here.

# COMMAND ----------

customers = spark.sql(
    """
    SELECT
      `Customer ID`                                            AS customer_id,
      FIRST(Country, TRUE)                                     AS country,
      SUM(line_revenue)                                        AS net_revenue,
      COUNT(DISTINCT CASE WHEN NOT is_cancel THEN Invoice END) AS sale_invoices,
      MIN(InvoiceDate)                                         AS first_seen,
      MAX(InvoiceDate)                                         AS last_seen
    FROM trade
    WHERE `Customer ID` IS NOT NULL
    GROUP BY `Customer ID`
    """
)
cust_w = Window.orderBy(F.desc("net_revenue"))
customers = customers.withColumn("rank", F.row_number().over(cust_w)).withColumn(
    "cum_share", F.sum("net_revenue").over(cust_w) / F.sum("net_revenue").over(Window.partitionBy())
)
display(customers.orderBy("rank").limit(10))

n_cust = customers.count()
print(f"Identified customers: {n_cust:,}")
print(f"One-invoice customers: {customers.filter('sale_invoices = 1').count():,}  ({customers.filter('sale_invoices = 1').count()/n_cust:.0%})")
print("Top 1% / 10% / 10 accounts cumulative share of identified net:")
for label, k in [("top 1%", max(n_cust // 100, 1)), ("top 10%", max(n_cust // 10, 1)), ("top 10", 10)]:
    share = customers.filter(F.col("rank") <= k).agg(F.max("cum_share")).first()[0]
    print(f"  {label}: {share:.1%}")

# COMMAND ----------

# Year-1 → year-2 retention on the same Dec–Nov windows
retention = spark.sql(
    """
    WITH y1 AS (
      SELECT DISTINCT `Customer ID` AS customer_id
      FROM sales
      WHERE `Customer ID` IS NOT NULL
        AND InvoiceDate >= '2009-12-01' AND InvoiceDate < '2010-12-01'
    ),
    y2 AS (
      SELECT DISTINCT `Customer ID` AS customer_id
      FROM sales
      WHERE `Customer ID` IS NOT NULL
        AND InvoiceDate >= '2010-12-01' AND InvoiceDate < '2011-12-01'
    )
    SELECT
      (SELECT COUNT(*) FROM y1)                         AS y1_accounts,
      (SELECT COUNT(*) FROM y2)                         AS y2_accounts,
      (SELECT COUNT(*) FROM y1 INNER JOIN y2 USING (customer_id)) AS retained,
      (SELECT COUNT(*) FROM y2 LEFT ANTI JOIN y1 USING (customer_id)) AS new_in_y2,
      (SELECT COUNT(*) FROM y1 LEFT ANTI JOIN y2 USING (customer_id)) AS lost
    """
)
display(retention)

# COMMAND ----------

# MAGIC %md
# MAGIC **Customer interpretation (fill)**
# MAGIC
# MAGIC - Concentration: top 10% of accounts = [ ] of identified net; top 10 accounts = [ ].
# MAGIC - Repeat vs one-off: [ ].
# MAGIC - Retention Dec–Nov: [rate], lost ≈ new? [ ]. Share of year-2 sales from retained accounts: [optional extra].
# MAGIC - Watch for a “top customer” that is really one cancelled mis-key (same pattern as §3.5).
# MAGIC - Confidence: [ ]. Caveat: anonymous sales are excluded here, so this is the account book, not the whole business.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Cancellations / returns, and geography

# COMMAND ----------

# Are the largest credits matched by a same-day sale of the same SKU and quantity?
big_credits = (
    returns.filter(F.col("line_revenue") < -50_000)
    .select("Invoice", "StockCode", "Quantity", PRICE_COL, "InvoiceDate", "Customer ID", "line_revenue")
)
print("Credit lines below −£50k (candidates for keyed-and-cancelled orders):")
display(big_credits)

display(
    spark.sql(
        """
        SELECT
          date_trunc('month', InvoiceDate) AS month,
          -SUM(line_revenue) / NULLIF(
            (SELECT SUM(line_revenue) FROM sales s
             WHERE date_trunc('month', s.InvoiceDate) = date_trunc('month', r.InvoiceDate)),
            0
          ) AS return_rate
        FROM returns r
        GROUP BY 1
        ORDER BY 1
        """
    )
)  # Visualisation: Bar, x=month, y=return_rate

# COMMAND ----------

geo = spark.sql(
    """
    SELECT
      Country,
      SUM(line_revenue)            AS net_revenue,
      COUNT(DISTINCT Invoice)      AS invoices,
      COUNT(DISTINCT `Customer ID`) AS identified_customers
    FROM sales
    GROUP BY Country
    """
)
geo = geo.withColumn("share", F.col("net_revenue") / F.sum("net_revenue").over(Window.partitionBy())).orderBy(F.desc("net_revenue"))
display(geo.limit(12))  # Visualisation: Bar, x=Country, y=net_revenue — hide United Kingdom to see the tail

# COMMAND ----------

# MAGIC %md
# MAGIC **Returns & geography interpretation (fill)**
# MAGIC
# MAGIC - Merchandise return rate on the §3 definition: [ ]. Raw-file rate if you include fee reversals will be higher — say what you used.
# MAGIC - Same-day giant cancels: [present / absent]; they [do / do not] dominate the return tail.
# MAGIC - UK share of merchandise sales: [ ]. Next countries and how many accounts sit under them: [ ].
# MAGIC - Confidence: [ ]. Caveats: credits are not matched to originating invoices except for the giant lines we inspected; `Country` is billing country.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key facts (paste into Section 1 after this cell runs)

# COMMAND ----------

key_facts = spark.sql(
    """
    SELECT
      (SELECT COUNT(*) FROM retail)                                              AS rows_after_overlap_drop,
      (SELECT COUNT(DISTINCT Invoice) FROM retail)                               AS invoices,
      (SELECT COUNT(DISTINCT `Customer ID`) FROM retail)                         AS identified_customers,
      (SELECT COUNT(DISTINCT Country) FROM retail)                               AS countries,
      (SELECT MIN(InvoiceDate) FROM retail)                                      AS date_min,
      (SELECT MAX(InvoiceDate) FROM retail)                                      AS date_max,
      (SELECT SUM(line_revenue) FROM sales)                                      AS merch_gross,
      (SELECT SUM(line_revenue) FROM returns)                                    AS merch_returns,
      (SELECT SUM(line_revenue) FROM trade)                                      AS merch_net,
      (SELECT -SUM(line_revenue) FROM returns)
        / (SELECT SUM(line_revenue) FROM sales)                                  AS merch_return_rate
    """
)
display(key_facts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Findings and hypotheses
# MAGIC
# MAGIC Three to five items. Each one needs evidence, business meaning, confidence, and a caveat. Replace the brackets from the cells above — do not leave invented numbers.

# COMMAND ----------

# MAGIC %md
# MAGIC **1. Grain / quality — [short finding]**
# MAGIC
# MAGIC - Evidence: [sheet overlap counts; null rates; C vs non-C negatives].
# MAGIC - Why it matters: [does a naïve union change Dec 2010 / YoY?].
# MAGIC - Confidence: [high / medium / low] because [ ].
# MAGIC - Caveat: [ ].
# MAGIC
# MAGIC **2. Revenue over time — [short finding]**
# MAGIC
# MAGIC - Evidence: [Dec–Nov YoY table; monthly chart].
# MAGIC - Why it matters: [growth vs price/mix vs seasonality].
# MAGIC - Confidence: [ ].
# MAGIC - Caveat: two seasons; Dec 2011 is partial; anonymous sales in/out of the definition.
# MAGIC
# MAGIC **3. Products — [short finding]**
# MAGIC
# MAGIC - Evidence: [top 10 / SKUs to 50%].
# MAGIC - Why it matters: [range, inventory, what a “top product” chart would get wrong if it used gross only].
# MAGIC - Confidence: [ ].
# MAGIC - Caveat: `NON_PRODUCT` is inferred from descriptions.
# MAGIC
# MAGIC **4. Customers — [short finding]**
# MAGIC
# MAGIC - Evidence: [top-decile share; retention table].
# MAGIC - Why it matters: [retention vs acquisition; wholesale vs consumer].
# MAGIC - Confidence: [ ].
# MAGIC - Caveat: no Customer ID on [x%] of rows.
# MAGIC
# MAGIC **5. Returns and geography — [short finding / hypothesis]**
# MAGIC
# MAGIC - Evidence: [return rate; giant matched cancels; UK share; accounts per country].
# MAGIC - Why it matters: [false “returns problem”; distributor vs many-retailer markets].
# MAGIC - Confidence: [ ].
# MAGIC - Caveat: [guest / DOTCOM postage as a possible second channel — only if the tables support it].

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Caveats
# MAGIC
# MAGIC - **Incomplete last month.** Data ends on [date_max from key_facts]. Dec 2011 is not a full month.
# MAGIC - **Sheet assembly.** If 1–9 Dec 2010 exists on both tabs, any analysis that concatenated them without checking is wrong for that week and for YoY.
# MAGIC - **Revenue definition is a choice.** Excluding postage, `M` (Manual), fees, £0 notes, and `A` bad-debt rows changes the total. The waterfall in §3 is the audit trail. A finance team could reasonably include shipping.
# MAGIC - **Returns are prefix-based, not matched.** Every `C` invoice is treated as a merchandise credit unless we inspected it. We did not systematically link credits back to the original sale.
# MAGIC - **Anonymous sales.** Missing `Customer ID` rows are in revenue and out of customer metrics. If they are a consumer/web channel, wholesale AOVs are slightly diluted.
# MAGIC - **No cost, margin, or currency conversion.** Values are GBP as invoiced. Nothing here is a profit statement.
# MAGIC - **Two seasonal cycles only.** Q4 shape is visible; it is a thin basis for a forecast.
# MAGIC - **Product names are messy.** Same `StockCode`, several `Description` spellings. We grouped by `StockCode`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Recommended next steps
# MAGIC
# MAGIC If this were a client project, in order:
# MAGIC
# MAGIC 1. **Confirm the file assembly and code book** — sheet overlap, meaning of `M` / `DOT` / `AMAZONFEE` / `A`, and whether blank Customer IDs are a channel.
# MAGIC 2. **Lock one “trade” definition** (the §3 filters) as a single SQL view the rest of the team uses. Not a platform rebuild — one agreed query.
# MAGIC 3. **Match credit notes to original invoices** (same customer, SKU, qty, within N days) so return rates by product/account are real.
# MAGIC 4. **Account retention / cadence** — the useful commercial follow-up if concentration and repeat rates are high.
# MAGIC 5. **Add cost/margin and a product hierarchy** before anyone ranks “best” products or customers.
# MAGIC 6. **Do not build a forecast yet** — two Q4s. Agree the decision (stock, cash, staffing) first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Use of GenAI
# MAGIC
# MAGIC I used an AI coding assistant (Cursor) as a pair programmer, not as the analyst of record.
# MAGIC
# MAGIC **Used for**
# MAGIC - Databricks notebook scaffolding (`# COMMAND ----------`, `display()`, Spark SQL drafts).
# MAGIC - A first list of DQ checks (nulls, prefixes, overlap, extremes) to make sure I did not skip an obvious one.
# MAGIC
# MAGIC **Changed or rejected**
# MAGIC - Rejected a medallion / multi-notebook / pipeline layout — the brief is one interview notebook.
# MAGIC - Rejected dropping all missing Customer IDs up front (common Kaggle pattern). That throws away a large sales block and hides the £0-note vs real-guest split.
# MAGIC - Rejected ranking products on positive quantity only. That would promote a keyed-and-cancelled order if one exists — hence net ranking on `trade`.
# MAGIC - Rejected treating every negative quantity as a return before checking `C` vs non-`C` and £0 stock notes.
# MAGIC
# MAGIC **How I will check the output**
# MAGIC - Every headline number in Sections 1 and 5 must appear in a table this notebook computes (`key_facts`, waterfall, YoY, concentration).
# MAGIC - Re-derive the sheet-overlap claim three ways if it appears (row count, invoice set, value).
# MAGIC - Eyeball the extreme-quantity rows rather than trusting `summary()`.
# MAGIC - If a claim cannot be seen in a table (e.g. “anonymous = consumer web”), it stays a hypothesis with medium/low confidence.
# MAGIC
# MAGIC _After the run: note anything the assistant drafted that the numbers contradicted._
