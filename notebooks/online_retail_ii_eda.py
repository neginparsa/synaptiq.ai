# Databricks notebook source
# MAGIC %md
# MAGIC # Online Retail II — Exploratory Data Analysis
# MAGIC
# MAGIC UK giftware retailer, invoice **line items**, Dec 2009 – Dec 2011.  
# MAGIC Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
# MAGIC
# MAGIC **Questions**
# MAGIC 1. Grain and quality
# MAGIC 2. Revenue over time
# MAGIC 3. Products that drive sales
# MAGIC 4. Customers that drive sales
# MAGIC 5. Cancellations / returns and geography
# MAGIC
# MAGIC Run top to bottom. Text in **[brackets]** is filled only after the cells that compute those numbers have run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC
# MAGIC All Unity Catalog objects this notebook needs are created here if they are missing:
# MAGIC
# MAGIC 1. Schema `workspace.default`
# MAGIC 2. Volume `workspace.default.my_files` (raw files)
# MAGIC 3. Table `workspace.default.online_retail_ii` (both year tabs + `source_sheet`)
# MAGIC
# MAGIC Later, §10 writes the small dashboard tables into the same schema. We do not create Bronze/Silver/Gold schemas.
# MAGIC
# MAGIC Upload the parquet or Excel under the volume path before the first run if the table does not exist. `Customer ID` has a space, so the source table uses Delta column mapping.

# COMMAND ----------

from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window

dbutils.widgets.text("catalog_name", "workspace")
dbutils.widgets.text("schema_name", "default")
dbutils.widgets.text("volume_name", "my_files")
dbutils.widgets.text("table_name", "workspace.default.online_retail_ii")
dbutils.widgets.text(
    "volume_parquet",
    "/Volumes/workspace/default/my_files/online_retail/online_retail_ii.parquet",
)
dbutils.widgets.text(
    "volume_xlsx",
    "/Volumes/workspace/default/my_files/online_retail/online_retail_II.xlsx",
)

CATALOG = dbutils.widgets.get("catalog_name")
SCHEMA = dbutils.widgets.get("schema_name")
VOLUME = dbutils.widgets.get("volume_name")
TABLE = dbutils.widgets.get("table_name")
VOLUME_PARQUET = dbutils.widgets.get("volume_parquet")
VOLUME_XLSX = dbutils.widgets.get("volume_xlsx")
UC_SCHEMA = f"{CATALOG}.{SCHEMA}"
UC_VOLUME = f"{CATALOG}.{SCHEMA}.{VOLUME}"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {UC_VOLUME}")
print("schema:", UC_SCHEMA)
print("volume:", UC_VOLUME)


def _create_source_table():
    # Column mapping: Delta otherwise rejects the space in `Customer ID`.
    ddl_props = """
        TBLPROPERTIES (
          'delta.minReaderVersion' = '2',
          'delta.minWriterVersion' = '5',
          'delta.columnMapping.mode' = 'name'
        )
    """
    if Path(VOLUME_PARQUET).exists():
        spark.sql(
            f"""
            CREATE TABLE {TABLE}
            {ddl_props}
            AS
            SELECT * FROM read_files('{VOLUME_PARQUET}', format => 'parquet')
            """
        )
        print("created", TABLE, "from", VOLUME_PARQUET)
        return
    if Path(VOLUME_XLSX).exists():
        import pandas as pd

        frames = []
        for sheet in ["Year 2009-2010", "Year 2010-2011"]:
            part = pd.read_excel(
                VOLUME_XLSX,
                sheet_name=sheet,
                dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
            )
            part["source_sheet"] = sheet
            frames.append(part)
        spark.createDataFrame(pd.concat(frames, ignore_index=True)).createOrReplaceTempView("_src")
        spark.sql(f"CREATE TABLE {TABLE} {ddl_props} AS SELECT * FROM _src")
        print("created", TABLE, "from", VOLUME_XLSX)
        return
    raise FileNotFoundError(
        f"Upload the dataset to {VOLUME_PARQUET} or {VOLUME_XLSX}, then re-run Setup."
    )


if spark.catalog.tableExists(TABLE):
    print("using existing", TABLE)
else:
    _create_source_table()

raw = spark.table(TABLE)
raw.createOrReplaceTempView("retail_raw")
PRICE_COL = "Price" if "Price" in raw.columns else "UnitPrice"
assert PRICE_COL in raw.columns, raw.columns
print("columns:", raw.columns)
print("price column (line_revenue is created after the overview tables):", PRICE_COL)
raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Executive summary
# MAGIC
# MAGIC _Fill from the `key_facts` table at the end of Section 4. Do not invent numbers._
# MAGIC
# MAGIC - **Grain:** [ ]
# MAGIC - **Coverage:** [n_rows] rows, [n_invoices] invoices, [n_customers] identified customers, [n_countries] countries, [date_min] → [date_max]
# MAGIC - **Merchandise revenue (definition in §3):** [gross] gross, [returns] returns, [net] net
# MAGIC - **Time:** [ ]
# MAGIC - **Customers / products:** [ ]
# MAGIC - **Returns / geography:** [ ]
# MAGIC - **Most trusted / least certain:** [ ]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data overview
# MAGIC
# MAGIC Confirm names, grain, and sheet coverage **before** creating `line_revenue`.

# COMMAND ----------

display(spark.sql(
    """
    SELECT
      COUNT(*)                                     AS n_rows,
      COUNT(DISTINCT Invoice)                      AS n_invoices,
      COUNT(DISTINCT StockCode)                    AS n_stock_codes,
      COUNT(DISTINCT `Customer ID`)                AS n_identified_customers,
      COUNT(DISTINCT Country)                      AS n_countries,
      MIN(InvoiceDate)                             AS date_min,
      MAX(InvoiceDate)                             AS date_max
    FROM retail_raw
    """
))

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

# (Invoice, StockCode) is not unique. That does not by itself identify a line —
# the file has no line-item id. Apparent grain: an invoice transaction line.
grain = raw.groupBy("Invoice", "StockCode").count()
display(
    grain.agg(
        F.count("*").alias("invoice_sku_pairs"),
        F.sum(F.when(F.col("count") > 1, 1)).alias("pairs_with_repeats"),
        F.max("count").alias("max_repeats"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Overview (fill after the tables)**
# MAGIC
# MAGIC - Price column name: **[Price / UnitPrice]** — `line_revenue` is created only after this.
# MAGIC - Grain: [invoice transaction line; (Invoice, StockCode) is / is not unique; no explicit line id].
# MAGIC - Sheet date ranges: [ ]. Overlap is a *suspect* until §3.2 proves it at transaction grain.

# COMMAND ----------

raw = raw.withColumn("line_revenue", F.col("Quantity") * F.col(PRICE_COL))
raw.createOrReplaceTempView("retail_raw")
print("price column:", PRICE_COL)
display(raw.limit(8))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data quality
# MAGIC
# MAGIC Investigate first. Do not drop rows because a pattern *looks* like a problem.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Nulls

# COMMAND ----------

display(spark.sql(
    f"""
    SELECT
      COUNT(*)                                          AS n_rows,
      SUM(CAST(Description IS NULL AS INT))             AS description_null,
      SUM(CAST(`Customer ID` IS NULL AS INT))           AS customer_id_null,
      ROUND(AVG(CAST(`Customer ID` IS NULL AS INT)), 4) AS customer_id_null_rate,
      ROUND(AVG(CAST(Description IS NULL AS INT)), 4)   AS description_null_rate,
      SUM(CAST(Invoice IS NULL OR StockCode IS NULL
               OR Quantity IS NULL OR InvoiceDate IS NULL
               OR `{PRICE_COL}` IS NULL OR Country IS NULL AS INT)) AS other_key_nulls
    FROM retail_raw
    """
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Duplicates and the suspected Dec-2010 sheet overlap
# MAGIC
# MAGIC Two different things:
# MAGIC 1. Exact duplicate *lines* inside a sheet (may be real extra scans).
# MAGIC 2. The same *transaction* stored on both year-tabs.
# MAGIC
# MAGIC **Do not drop the Dec-2010 window until the join below shows a transaction-level match.** Calendar overlap alone is not enough.

# COMMAND ----------

KEY = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", PRICE_COL, "Customer ID", "Country"]

print("surplus exact copies (including source_sheet):", raw.count() - raw.dropDuplicates(KEY + ["source_sheet"]).count())
print("surplus exact copies (ignoring source_sheet): ", raw.count() - raw.dropDuplicates(KEY).count())

# Candidate window from the sheet date ranges — still only a hypothesis
s1 = raw.filter((F.col("source_sheet").contains("2009")) & (F.col("InvoiceDate") >= "2010-12-01"))
s2 = raw.filter((F.col("source_sheet").contains("2010-2011")) & (F.col("InvoiceDate") < "2010-12-10"))

s1_n, s2_n = s1.count(), s2.count()
s1_val = s1.agg(F.sum("line_revenue")).first()[0]
s2_val = s2.agg(F.sum("line_revenue")).first()[0]

matched = s1.join(s2, on=KEY, how="inner").count()
only_s1 = s1.join(s2, on=KEY, how="left_anti").count()
only_s2 = s2.join(s1, on=KEY, how="left_anti").count()

overlap_proof = spark.createDataFrame(
    [
        ("sheet-1 rows dated ≥ 2010-12-01", s1_n, float(s1_val or 0)),
        ("sheet-2 rows dated 1–9 Dec 2010", s2_n, float(s2_val or 0)),
        ("inner join on full line-item key", matched, None),
        ("in sheet-1 window, not in sheet-2", only_s1, None),
        ("in sheet-2 window, not in sheet-1", only_s2, None),
    ],
    ["check", "rows", "line_revenue"],
)
display(overlap_proof)

overlap_proven = s1_n > 0 and s1_n == s2_n == matched and only_s1 == 0 and only_s2 == 0
print("transaction-level overlap proven:" , overlap_proven)

# COMMAND ----------

# MAGIC %md
# MAGIC **Overlap decision.** If every line in the sheet-1 window joins to an identical line in the sheet-2 window on the full key, a naïve union double-counts those invoices. Only then drop the *sheet-1 copy*. If the join is incomplete, keep both and treat “overlap” as unproven.
# MAGIC
# MAGIC Same-invoice repeated lines are **not** dropped here.

# COMMAND ----------

if overlap_proven:
    df = raw.filter(~((F.col("source_sheet").contains("2009")) & (F.col("InvoiceDate") >= "2010-12-01")))
    print(f"Dropped sheet-1 copy of the matched window: {raw.count():,} → {df.count():,}")
else:
    df = raw
    print("Overlap not proven at transaction grain — keeping all rows.")

df.createOrReplaceTempView("retail")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Negative quantities, invoice prefixes, prices
# MAGIC
# MAGIC Do **not** treat every negative quantity as a return. `C` is the usual credit prefix on this file; check whether that is the whole story.

# COMMAND ----------

df = df.withColumn(
    "invoice_prefix",
    F.when(F.regexp_extract("Invoice", r"^([A-Za-z]+)", 1) == "", F.lit("(numeric)")).otherwise(
        F.regexp_extract("Invoice", r"^([A-Za-z]+)", 1)
    ),
)
df.createOrReplaceTempView("retail")

display(spark.sql(
    f"""
    SELECT
      invoice_prefix,
      COUNT(*)                        AS rows,
      COUNT(DISTINCT Invoice)         AS invoices,
      SUM(CAST(Quantity < 0 AS INT))  AS neg_qty_rows,
      SUM(CAST(`{PRICE_COL}` < 0 AS INT)) AS neg_price_rows,
      SUM(CAST(`{PRICE_COL}` = 0 AS INT)) AS zero_price_rows,
      SUM(line_revenue)               AS net_line_revenue
    FROM retail
    GROUP BY invoice_prefix
    ORDER BY rows DESC
    """
))

neg_not_c = df.filter((F.col("Quantity") < 0) & (F.col("invoice_prefix") == "(numeric)"))
print("neg qty on non-C invoices:", neg_not_c.count())
if neg_not_c.count() > 0:
    print(
        "  share price==0:",
        f"{neg_not_c.filter(F.col(PRICE_COL) == 0).count() / neg_not_c.count():.1%}",
        "  share missing customer:",
        f"{neg_not_c.filter(F.col('Customer ID').isNull()).count() / neg_not_c.count():.1%}",
    )

display(
    df.filter(F.col(PRICE_COL) < 0)
    .select("Invoice", "StockCode", "Description", "Quantity", PRICE_COL, "Customer ID", "InvoiceDate")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Missing Customer ID

# COMMAND ----------

guest = df.filter(F.col("Customer ID").isNull())
guest_inv = guest.groupBy("Invoice").agg(
    (F.sum(F.when(F.col(PRICE_COL) == 0, 1).otherwise(0)) == F.count("*")).alias("all_zero_price")
)
print(f"rows with no Customer ID: {guest.count():,} ({guest.count() / df.count():.1%})")
print(f"invoices with no Customer ID: {guest_inv.count():,}")
print("  entirely £0:", guest_inv.filter("all_zero_price").count())
print("  not entirely £0:", guest_inv.filter("NOT all_zero_price").count())

display(
    guest.filter(F.col(PRICE_COL) == 0)
    .groupBy("Description")
    .count()
    .orderBy(F.desc("count"))
    .limit(12)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Extremes and non-product stock codes

# COMMAND ----------

display(df.select("Quantity", PRICE_COL, "line_revenue").summary())

display(
    df.orderBy(F.desc(F.abs("Quantity")))
    .select("Invoice", "StockCode", "Description", "Quantity", PRICE_COL, "InvoiceDate", "Customer ID")
    .limit(12)
)

display(
    df.filter(F.col("StockCode").rlike("^[A-Za-z]"))
    .groupBy("StockCode")
    .agg(
        F.count("*").alias("rows"),
        F.first("Description", ignorenulls=True).alias("example_description"),
        F.sum("line_revenue").alias("net_revenue"),
    )
    .orderBy(F.desc("rows"))
    .limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Time coverage

# COMMAND ----------

inv = df.groupBy("Invoice").agg(F.min("InvoiceDate").alias("ts"))
display(inv.groupBy(F.date_format("ts", "E").alias("dow")).count().orderBy("dow"))
display(inv.groupBy(F.hour("ts").alias("hour")).count().orderBy("hour"))

last = df.filter(F.col("InvoiceDate") >= "2011-12-01")
print("max timestamp:", df.agg(F.max("InvoiceDate")).first()[0])
print("distinct days in Dec 2011:", last.select(F.dayofmonth("InvoiceDate")).distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC **DQ notes (fill after the tables)**
# MAGIC
# MAGIC | Check | What I saw | Decision |
# MAGIC |---|---|---|
# MAGIC | Sheet overlap | [proven / not proven] | Drop sheet-1 copy only if the join matched 1:1 |
# MAGIC | Duplicate lines | [ ] | Keep unless value is material |
# MAGIC | `C` vs other neg qty | [ ] | Treat `C` as credits; do not assume the rest are returns |
# MAGIC | Neg / zero price | [ ] | Hold out of merchandise revenue if they are adjustments / notes |
# MAGIC | Missing Customer ID | [ ] | Keep in revenue; exclude from customer metrics |
# MAGIC | Extreme qty | [ ] | Look for a same-day `C` before ranking products |
# MAGIC | Alpha stock codes | [read the §3.5 table] | Then fill `NON_PRODUCT_CODES` below — do not guess |
# MAGIC | Last month | [ ] | Do not compare a partial December to a full month |

# COMMAND ----------

# Fill AFTER reading the alpha StockCode table in §3.5. Start empty so nothing
# is excluded on an uninspected regex. Add a code only if you can say what it is.
#   POST      — if description is postage
#   DOT       — if description is DOTCOM postage
#   M         — Manual; inspect before excluding (can be real sales or adjustments)
#   AMAZONFEE — marketplace fee
#   C2 / D / S / BANK CHARGES / CRUK / gift_* / B — only if the table supports it
NON_PRODUCT_CODES = [
    # "POST",
    # "DOT",
]

is_non_product = (
    F.col("StockCode").isin(NON_PRODUCT_CODES) if NON_PRODUCT_CODES else F.lit(False)
)
df = (
    df.withColumn("is_cancel", F.col("invoice_prefix") == "C")
    .withColumn("is_adjust", F.col("invoice_prefix") == "A")
    .withColumn("is_non_product", is_non_product)
    .withColumn("is_zero_price", F.col(PRICE_COL) == 0)
)

trade = df.filter(~F.col("is_adjust") & ~F.col("is_zero_price") & ~F.col("is_non_product"))
sales = trade.filter(~F.col("is_cancel") & (F.col("Quantity") > 0))
returns = trade.filter(F.col("is_cancel"))  # C-invoice lines; negatives stay in

sales.createOrReplaceTempView("sales")
returns.createOrReplaceTempView("returns")
trade.createOrReplaceTempView("trade")

display(spark.createDataFrame(
    [
        ("raw table", raw.count(), float(raw.agg(F.sum("line_revenue")).first()[0] or 0)),
        ("after overlap decision", df.count(), float(df.agg(F.sum("line_revenue")).first()[0] or 0)),
        ("merchandise trade", trade.count(), float(trade.agg(F.sum("line_revenue")).first()[0] or 0)),
        ("  sales", sales.count(), float(sales.agg(F.sum("line_revenue")).first()[0] or 0)),
        ("  C-invoice credits", returns.count(), float(returns.agg(F.sum("line_revenue")).first()[0] or 0)),
    ],
    ["step", "rows", "net_line_revenue"],
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Exploratory analysis
# MAGIC
# MAGIC Use `display()` → **Line** / **Bar** on the result.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Revenue over time
# MAGIC
# MAGIC If Dec 2011 is partial, compare **Dec–Nov vs Dec–Nov**, not calendar years.

# COMMAND ----------

display(spark.sql(
    """
    SELECT
      date_trunc('month', InvoiceDate) AS month,
      SUM(CASE WHEN Quantity > 0 THEN line_revenue ELSE 0 END) AS gross,
      SUM(CASE WHEN Quantity < 0 THEN line_revenue ELSE 0 END) AS returns,
      SUM(line_revenue) AS net,
      COUNT(DISTINCT Invoice) AS invoices
    FROM trade
    GROUP BY 1
    ORDER BY 1
    """
))  # Line: x=month, y=gross / net

display(spark.sql(
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
      SUM(CASE WHEN is_cancel THEN line_revenue END) AS returns,
      SUM(line_revenue) AS net,
      COUNT(DISTINCT CASE WHEN Quantity > 0 AND NOT is_cancel THEN Invoice END) AS sale_invoices,
      COUNT(DISTINCT `Customer ID`) AS identified_customers,
      SUM(CASE WHEN Quantity > 0 AND NOT is_cancel THEN Quantity END) AS units
    FROM tagged
    WHERE window IS NOT NULL
    GROUP BY window
    ORDER BY window
    """
))

# COMMAND ----------

# MAGIC %md
# MAGIC **Time (fill):** seasonality [ ]; Dec–Nov gross/net/invoices/units [ ]; Dec 2011 [partial / full]. Confidence [ ]. Caveat [ ].

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Products
# MAGIC
# MAGIC Rank on **net** merchandise so a same-day cancel cannot become a top product.

# COMMAND ----------

products = spark.sql(
    """
    SELECT StockCode,
           FIRST(Description, TRUE) AS description,
           SUM(line_revenue) AS net_revenue,
           SUM(Quantity) AS net_units,
           COUNT(DISTINCT Invoice) AS invoices
    FROM trade
    GROUP BY StockCode
    """
).filter("net_revenue > 0")

prod_total = products.agg(F.sum("net_revenue")).first()[0]
w = Window.partitionBy(F.lit(1)).orderBy(F.desc("net_revenue"))
products = products.withColumn("rank", F.row_number().over(w)).withColumn(
    "cum_share", F.sum("net_revenue").over(w) / F.lit(prod_total)
)
display(products.limit(15))  # Bar: x=description, y=net_revenue

print("SKUs with positive net:", products.count())
print("top 10 share:", products.filter("rank <= 10").agg(F.max("cum_share")).first()[0])
print("SKUs to ~50% / ~80%:", products.filter("cum_share <= 0.50").count(), "/", products.filter("cum_share <= 0.80").count())

# COMMAND ----------

# MAGIC %md
# MAGIC **Products (fill):** top SKU and share [ ]; tail [ ]; any “top product” that vanished after netting credits [ ]. Confidence [ ]. Caveat: only codes listed in `NON_PRODUCT_CODES`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Customers
# MAGIC
# MAGIC Identified accounts only. Anonymous lines stay in §4.1 revenue and are out of this section.

# COMMAND ----------

customers = spark.sql(
    """
    SELECT
      `Customer ID` AS customer_id,
      FIRST(Country, TRUE) AS country,
      SUM(line_revenue) AS net_revenue,
      COUNT(DISTINCT CASE WHEN NOT is_cancel THEN Invoice END) AS sale_invoices,
      MIN(InvoiceDate) AS first_seen,
      MAX(InvoiceDate) AS last_seen
    FROM trade
    WHERE `Customer ID` IS NOT NULL
    GROUP BY `Customer ID`
    """
)
cust_total = customers.agg(F.sum("net_revenue")).first()[0]
cw = Window.partitionBy(F.lit(1)).orderBy(F.desc("net_revenue"))
customers = customers.withColumn("rank", F.row_number().over(cw)).withColumn(
    "cum_share", F.sum("net_revenue").over(cw) / F.lit(cust_total)
)
display(customers.orderBy("rank").limit(10))

n = customers.count()
print(f"identified customers: {n:,}")
print(f"one-invoice: {customers.filter('sale_invoices = 1').count():,} ({customers.filter('sale_invoices = 1').count() / n:.0%})")
for label, k in [("top 1%", max(n // 100, 1)), ("top 10%", max(n // 10, 1)), ("top 10", 10)]:
    print(f"  {label}: {customers.filter(F.col('rank') <= k).agg(F.max('cum_share')).first()[0]:.1%}")

display(spark.sql(
    """
    WITH y1 AS (
      SELECT DISTINCT `Customer ID` AS id FROM sales
      WHERE `Customer ID` IS NOT NULL
        AND InvoiceDate >= '2009-12-01' AND InvoiceDate < '2010-12-01'
    ),
    y2 AS (
      SELECT DISTINCT `Customer ID` AS id FROM sales
      WHERE `Customer ID` IS NOT NULL
        AND InvoiceDate >= '2010-12-01' AND InvoiceDate < '2011-12-01'
    )
    SELECT
      (SELECT COUNT(*) FROM y1) AS y1_accounts,
      (SELECT COUNT(*) FROM y2) AS y2_accounts,
      (SELECT COUNT(*) FROM y1 JOIN y2 USING (id)) AS retained,
      (SELECT COUNT(*) FROM y2 LEFT ANTI JOIN y1 USING (id)) AS new_in_y2,
      (SELECT COUNT(*) FROM y1 LEFT ANTI JOIN y2 USING (id)) AS lost
    """
))

# COMMAND ----------

# MAGIC %md
# MAGIC **Customers (fill):** top 10% / top 10 share [ ]; one-invoice [ ]; retention [ ]; any top account that is a cancelled mis-key [ ]. Confidence [ ]. Caveat: no Customer ID on [x%] of rows.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Cancellations / returns and geography

# COMMAND ----------

display(
    returns.filter(F.col("line_revenue") < -50_000)
    .select("Invoice", "StockCode", "Quantity", PRICE_COL, "InvoiceDate", "Customer ID", "line_revenue")
)

display(spark.sql(
    """
    WITH monthly_returns AS (
      SELECT date_trunc('month', InvoiceDate) AS month,
             -SUM(line_revenue) AS return_value
      FROM returns
      GROUP BY 1
    ),
    monthly_sales AS (
      SELECT date_trunc('month', InvoiceDate) AS month,
             SUM(line_revenue) AS gross_sales
      FROM sales
      GROUP BY 1
    )
    SELECT
      r.month,
      r.return_value,
      s.gross_sales,
      r.return_value / NULLIF(s.gross_sales, 0) AS return_rate
    FROM monthly_returns r
    LEFT JOIN monthly_sales s USING (month)
    ORDER BY month
    """
))  # Bar: x=month, y=return_rate

geo = spark.sql(
    """
    SELECT Country,
           SUM(line_revenue) AS net_revenue,
           COUNT(DISTINCT Invoice) AS invoices,
           COUNT(DISTINCT `Customer ID`) AS identified_customers
    FROM sales
    GROUP BY Country
    """
)
geo_total = geo.agg(F.sum("net_revenue")).first()[0]
geo = geo.withColumn("share", F.col("net_revenue") / F.lit(geo_total))
display(geo.orderBy(F.desc("net_revenue")).limit(12))  # Bar: hide UK to see the tail

# COMMAND ----------

# MAGIC %md
# MAGIC **Returns & geography (fill):** merchandise return rate [ ]; giant credits [ ]; UK share [ ]; next countries and account counts [ ]. Confidence [ ]. Caveat: credits are not matched to original invoices except the lines we inspect.

# COMMAND ----------

display(spark.sql(
    """
    SELECT
      (SELECT COUNT(*) FROM retail)                      AS rows_after_overlap_decision,
      (SELECT COUNT(DISTINCT Invoice) FROM retail)       AS invoices,
      (SELECT COUNT(DISTINCT `Customer ID`) FROM retail) AS identified_customers,
      (SELECT COUNT(DISTINCT Country) FROM retail)       AS countries,
      (SELECT MIN(InvoiceDate) FROM retail)              AS date_min,
      (SELECT MAX(InvoiceDate) FROM retail)              AS date_max,
      (SELECT SUM(line_revenue) FROM sales)              AS merch_gross,
      (SELECT SUM(line_revenue) FROM returns)            AS merch_returns,
      (SELECT SUM(line_revenue) FROM trade)              AS merch_net,
      (SELECT -SUM(line_revenue) FROM returns)
        / (SELECT SUM(line_revenue) FROM sales)          AS merch_return_rate
    """
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Findings and hypotheses
# MAGIC
# MAGIC Replace brackets from the tables above. Do not invent numbers.
# MAGIC
# MAGIC **1. Grain / quality — [ ]**
# MAGIC - Evidence: [overlap join; null rates; C vs non-C negatives]
# MAGIC - Why it matters: [ ]
# MAGIC - Confidence: [ ] because [ ]
# MAGIC - Caveat: [ ]
# MAGIC
# MAGIC **2. Revenue over time — [ ]**
# MAGIC - Evidence: [YoY table; monthly chart]
# MAGIC - Why it matters: [ ]
# MAGIC - Confidence: [ ]
# MAGIC - Caveat: two seasons; last month may be partial
# MAGIC
# MAGIC **3. Products — [ ]**
# MAGIC - Evidence: [top 10 / SKUs to 50%]
# MAGIC - Why it matters: [ ]
# MAGIC - Confidence: [ ]
# MAGIC - Caveat: only codes you listed in `NON_PRODUCT_CODES` after §3.5
# MAGIC
# MAGIC **4. Customers — [ ]**
# MAGIC - Evidence: [concentration; retention]
# MAGIC - Why it matters: [ ]
# MAGIC - Confidence: [ ]
# MAGIC - Caveat: missing Customer ID on [x%] of rows
# MAGIC
# MAGIC **5. Returns and geography — [ ]**
# MAGIC - Evidence: [return rate; large credits; country table]
# MAGIC - Why it matters: [ ]
# MAGIC - Confidence: [ ]
# MAGIC - Caveat: [ ]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Caveats
# MAGIC
# MAGIC - **Last month may be incomplete.** Use `date_max` from `key_facts` before comparing Decembers.
# MAGIC - **Overlap is conditional.** Rows were removed only if §3.2 proved a 1:1 key match. Calendar overlap by itself was not treated as proof.
# MAGIC - **Revenue definition is a choice.** `A` rows and £0 notes are out of `trade`. Other exclusions are only the StockCodes you listed after §3.5. The waterfall is the audit trail.
# MAGIC - **Returns are prefix-based.** `C` invoices are credits; they are not matched to originating sales except the large lines we inspect.
# MAGIC - **Anonymous sales** are in revenue and out of customer metrics.
# MAGIC - **No cost or margin.** GBP as invoiced. Not a profit statement.
# MAGIC - **Two seasonal cycles.** Thin basis for a forecast.
# MAGIC - **Descriptions are messy.** Grouped by `StockCode`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Recommended next steps
# MAGIC
# MAGIC 1. Confirm the year-tab assembly, the code book (`M`, `DOT`, `AMAZONFEE`, `A`), and what blank Customer IDs are.
# MAGIC 2. Lock the §3 `trade` filters as one shared SQL view — not a platform rebuild.
# MAGIC 3. Match credit notes to original invoices (same customer, SKU, qty, within N days).
# MAGIC 4. Account retention / order cadence if concentration and repeat rates are high.
# MAGIC 5. Add cost/margin and a product hierarchy before ranking “best” products or customers.
# MAGIC 6. Do not forecast until the decision (stock, cash, staffing) is clear — two Q4s only.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Use of GenAI
# MAGIC
# MAGIC AI assistant used as a pair programmer, not the analyst of record.
# MAGIC
# MAGIC **Used for:** Databricks cell layout, a first DQ checklist, Spark SQL drafts.
# MAGIC
# MAGIC **Rejected / corrected:**
# MAGIC - Medallion/pipeline layout; dropping all missing Customer IDs; ranking products on positive quantity only; treating every negative qty as a return; dropping the Dec-2010 window from date overlap alone.
# MAGIC - A first-draft monthly return-rate query joined monthly sales back onto *line-level* returns, so `SUM(sales)` could multiply the denominator by the number of return rows. Rewrote it to aggregate returns and sales by month first, then join.
# MAGIC - A pre-baked `NON_PRODUCT` regex. Codes are excluded only after the §3.5 table, and only if I can name what each code is.
# MAGIC
# MAGIC **Checks:** every number in §§1 and 5 must appear in a table this notebook computes; overlap must be a full-key join, not a date-range count; SQL joins are checked at the grain of the aggregation; extreme-qty rows are inspected, not trusted from `summary()`; claims without a table stay hypotheses.
# MAGIC
# MAGIC _After the run: note anything the assistant drafted that the numbers contradicted._

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. From EDA to a consumption layer (not a rebuild)
# MAGIC
# MAGIC This exercise stops at a notebook plus a thin dashboard. The path is:
# MAGIC
# MAGIC ```
# MAGIC Raw dataset
# MAGIC     ↓
# MAGIC Data quality / validation
# MAGIC     ↓
# MAGIC trade / sales / returns   ← the only metric definitions
# MAGIC     ↓
# MAGIC EDA notebook              ← the analytical artifact
# MAGIC     ↓
# MAGIC Dashboard views           ← same definitions, no new logic
# MAGIC ```
# MAGIC
# MAGIC There is no Bronze / Silver / Gold here. The EDA is what would *inform* a later production design:
# MAGIC
# MAGIC ```
# MAGIC Sources → Bronze → Silver (canonical transactions) → Gold (governed KPIs)
# MAGIC                                              ↓
# MAGIC                         Databricks SQL / Dashboard / Genie
# MAGIC ```
# MAGIC
# MAGIC Unity Catalog would then own access, lineage, and discovery. Silver would need the classifications this notebook already found it must distinguish: **SALE**, **CREDIT**, **ADJUSTMENT**, **FEE / NON-MERCHANDISE**, **UNKNOWN**.
# MAGIC
# MAGIC Principle: the EDA informs the architecture; the architecture is not imposed on a 2–3 hour exploratory exercise.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Dashboard queries
# MAGIC
# MAGIC Consumption only. Every metric reads `trade` / `sales` / `returns` from §3. No new filters.
# MAGIC
# MAGIC | Query | Chart |
# MAGIC |---|---|
# MAGIC | `dashboard_kpis` | KPI row |
# MAGIC | `dashboard_monthly_revenue` | Line: x=month, y=gross_revenue and net_revenue. Treat `is_partial_period` as a warning, not a decline. |
# MAGIC | `dashboard_top_products` | Bar: category=description, value=net_revenue |
# MAGIC | `dashboard_top_customers` | Bar: category=customer_id, value=net_revenue |
# MAGIC | `dashboard_geography` | Full country mix (includes UK) |
# MAGIC | `dashboard_geography_ex_uk` | Bar for the chart only — UK removed so the tail is readable |
# MAGIC | `dashboard_monthly_returns` | Line: x=month, y=return_rate (percent). Built from two monthly aggregates, then joined. |
# MAGIC
# MAGIC After this cell runs, the same result sets are written as tables in `workspace.default` (the schema created in Setup). Point the Databricks dashboard at those tables. Do not type KPI numbers by hand.

# COMMAND ----------

# Same definitions as §3. Re-declared only so this section is readable; the filters do not change.
sales.createOrReplaceTempView("sales")
returns.createOrReplaceTempView("returns")
trade.createOrReplaceTempView("trade")

spark.sql(
    """
    CREATE OR REPLACE TEMP VIEW dashboard_kpis AS
    SELECT
      -- net merchandise = validated trade (sales + C-invoice credits)
      (SELECT SUM(line_revenue) FROM trade) AS net_merchandise_revenue,
      -- gross = positive merchandise sales only
      (SELECT SUM(line_revenue) FROM sales) AS gross_merchandise_revenue,
      -- return/credit value = absolute C-invoice merchandise credits
      (SELECT -SUM(line_revenue) FROM returns) AS return_credit_value,
      (SELECT COUNT(DISTINCT Invoice) FROM sales) AS sale_invoices,
      (SELECT COUNT(DISTINCT `Customer ID`) FROM sales) AS identified_customers,
      (SELECT -SUM(line_revenue) FROM returns)
        / (SELECT SUM(line_revenue) FROM sales) AS return_credit_rate
    """
)

# Monthly grain from sales + returns, then net = gross - credits.
# is_partial_period flags Dec 2011 because the source file ends on the 9th — not a decline.
spark.sql(
    """
    CREATE OR REPLACE TEMP VIEW dashboard_monthly_revenue AS
    WITH monthly_sales AS (
      SELECT date_trunc('month', InvoiceDate) AS month,
             SUM(line_revenue) AS gross_revenue,
             COUNT(DISTINCT Invoice) AS sale_invoices
      FROM sales
      GROUP BY 1
    ),
    monthly_returns AS (
      SELECT date_trunc('month', InvoiceDate) AS month,
             -SUM(line_revenue) AS return_value
      FROM returns
      GROUP BY 1
    )
    SELECT
      s.month,
      s.gross_revenue,
      COALESCE(r.return_value, 0) AS return_value,
      s.gross_revenue - COALESCE(r.return_value, 0) AS net_revenue,
      s.sale_invoices,
      (s.month = TIMESTAMP '2011-12-01') AS is_partial_period
    FROM monthly_sales s
    LEFT JOIN monthly_returns r USING (month)
    ORDER BY month
    """
)

# Net by StockCode so a same-day credit cannot rank as a top product.
spark.sql(
    """
    CREATE OR REPLACE TEMP VIEW dashboard_top_products AS
    SELECT
      StockCode,
      FIRST(Description, TRUE) AS description,
      SUM(line_revenue) AS net_revenue,
      SUM(Quantity) AS net_units,
      COUNT(DISTINCT Invoice) AS invoices
    FROM trade
    GROUP BY StockCode
    ORDER BY net_revenue DESC
    LIMIT 10
    """
)

# Identified accounts only. Anonymous sales stay in dashboard_kpis / monthly revenue.
spark.sql(
    """
    CREATE OR REPLACE TEMP VIEW dashboard_top_customers AS
    SELECT
      `Customer ID` AS customer_id,
      FIRST(Country, TRUE) AS country,
      SUM(line_revenue) AS net_revenue,
      COUNT(DISTINCT CASE WHEN NOT is_cancel THEN Invoice END) AS sale_invoices
    FROM trade
    WHERE `Customer ID` IS NOT NULL
    GROUP BY `Customer ID`
    ORDER BY net_revenue DESC
    LIMIT 10
    """
)

spark.sql(
    """
    CREATE OR REPLACE TEMP VIEW dashboard_geography AS
    WITH totals AS (SELECT SUM(line_revenue) AS all_rev FROM sales)
    SELECT
      Country,
      SUM(s.line_revenue) AS net_revenue,
      COUNT(DISTINCT Invoice) AS invoices,
      COUNT(DISTINCT `Customer ID`) AS identified_customers,
      SUM(s.line_revenue) / FIRST(t.all_rev) AS revenue_share
    FROM sales s
    CROSS JOIN totals t
    GROUP BY Country
    """
)

# Visualization helper only. UK remains in dashboard_geography and in the KPIs.
spark.sql(
    """
    CREATE OR REPLACE TEMP VIEW dashboard_geography_ex_uk AS
    SELECT * FROM dashboard_geography
    WHERE Country <> 'United Kingdom'
    ORDER BY net_revenue DESC
    LIMIT 10
    """
)

# Aggregate each side to month FIRST, then join. Do not join monthly sales onto return lines.
spark.sql(
    """
    CREATE OR REPLACE TEMP VIEW dashboard_monthly_returns AS
    WITH monthly_sales AS (
      SELECT date_trunc('month', InvoiceDate) AS month,
             SUM(line_revenue) AS gross_sales
      FROM sales
      GROUP BY 1
    ),
    monthly_returns AS (
      SELECT date_trunc('month', InvoiceDate) AS month,
             -SUM(line_revenue) AS return_value
      FROM returns
      GROUP BY 1
    )
    SELECT
      s.month,
      s.gross_sales,
      COALESCE(r.return_value, 0) AS return_value,
      COALESCE(r.return_value, 0) / NULLIF(s.gross_sales, 0) AS return_rate,
      (s.month = TIMESTAMP '2011-12-01') AS is_partial_period
    FROM monthly_sales s
    LEFT JOIN monthly_returns r USING (month)
    ORDER BY month
    """
)

# Persist the same result sets as tables in the schema created in Setup.
# These are consumption copies of the temp views, not a second metric layer.
DASHBOARD_TABLES = [
    "dashboard_kpis",
    "dashboard_monthly_revenue",
    "dashboard_top_products",
    "dashboard_top_customers",
    "dashboard_geography",
    "dashboard_geography_ex_uk",
    "dashboard_monthly_returns",
]
for name in DASHBOARD_TABLES:
    spark.sql(f"CREATE OR REPLACE TABLE {UC_SCHEMA}.{name} AS SELECT * FROM {name}")
    print("wrote", f"{UC_SCHEMA}.{name}")

print("dashboard views ready")
display(spark.table("dashboard_kpis"))
display(spark.table("dashboard_monthly_revenue"))
display(spark.table("dashboard_top_products"))
display(spark.table("dashboard_top_customers"))
display(spark.table("dashboard_geography").orderBy(F.desc("net_revenue")).limit(15))
display(spark.table("dashboard_geography_ex_uk"))
display(spark.table("dashboard_monthly_returns"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dashboard reconciliation
# MAGIC
# MAGIC Compare each dashboard total to the notebook definition. Customer-level revenue **will not** equal merchandise net when `Customer ID` is missing — that is expected, not a break.

# COMMAND ----------

display(spark.sql(
    """
    WITH
    nb AS (
      SELECT
        (SELECT SUM(line_revenue) FROM trade) AS merch_net,
        (SELECT SUM(line_revenue) FROM sales) AS merch_gross,
        (SELECT -SUM(line_revenue) FROM returns) AS merch_returns,
        (SELECT COUNT(DISTINCT Invoice) FROM sales) AS sale_invoices,
        (SELECT COUNT(DISTINCT `Customer ID`) FROM sales) AS identified_customers,
        (SELECT SUM(line_revenue) FROM trade WHERE `Customer ID` IS NOT NULL) AS identified_net,
        (SELECT SUM(line_revenue) FROM sales WHERE Country <> 'United Kingdom') AS sales_ex_uk
    ),
    dash AS (
      SELECT
        (SELECT net_merchandise_revenue FROM dashboard_kpis) AS merch_net,
        (SELECT gross_merchandise_revenue FROM dashboard_kpis) AS merch_gross,
        (SELECT return_credit_value FROM dashboard_kpis) AS merch_returns,
        (SELECT sale_invoices FROM dashboard_kpis) AS sale_invoices,
        (SELECT identified_customers FROM dashboard_kpis) AS identified_customers,
        (SELECT SUM(net_revenue) FROM dashboard_monthly_revenue) AS monthly_net,
        (SELECT SUM(gross_revenue) FROM dashboard_monthly_revenue) AS monthly_gross,
        (SELECT SUM(return_value) FROM dashboard_monthly_revenue) AS monthly_returns,
        (SELECT SUM(net_revenue) FROM dashboard_geography) AS geo_sales,
        (SELECT SUM(net_revenue) FROM dashboard_geography WHERE Country <> 'United Kingdom') AS geo_ex_uk,
        (SELECT SUM(line_revenue) FROM trade WHERE `Customer ID` IS NOT NULL) AS identified_net
    )
    SELECT stack(10,
      'net merchandise revenue',        CAST(nb.merch_net AS DOUBLE),              CAST(dash.merch_net AS DOUBLE),
      'gross merchandise revenue',      CAST(nb.merch_gross AS DOUBLE),            CAST(dash.merch_gross AS DOUBLE),
      'return / credit value',          CAST(nb.merch_returns AS DOUBLE),          CAST(dash.merch_returns AS DOUBLE),
      'sale invoices',                  CAST(nb.sale_invoices AS DOUBLE),          CAST(dash.sale_invoices AS DOUBLE),
      'identified customers',           CAST(nb.identified_customers AS DOUBLE),   CAST(dash.identified_customers AS DOUBLE),
      'monthly net sums to trade net',  CAST(nb.merch_net AS DOUBLE),              CAST(dash.monthly_net AS DOUBLE),
      'monthly gross sums to sales',    CAST(nb.merch_gross AS DOUBLE),            CAST(dash.monthly_gross AS DOUBLE),
      'monthly returns sum to credits', CAST(nb.merch_returns AS DOUBLE),          CAST(dash.monthly_returns AS DOUBLE),
      'geography sums to sales',        CAST(nb.merch_gross AS DOUBLE),            CAST(dash.geo_sales AS DOUBLE),
      'non-UK geography vs non-UK sales', CAST(nb.sales_ex_uk AS DOUBLE),           CAST(dash.geo_ex_uk AS DOUBLE)
    ) AS (metric, notebook_value, dashboard_value)
    FROM nb CROSS JOIN dash
    """
).select(
    "metric",
    "notebook_value",
    "dashboard_value",
    (F.col("dashboard_value") - F.col("notebook_value")).alias("difference"),
    F.when(F.abs(F.col("dashboard_value") - F.col("notebook_value")) < 0.01, "reconcile")
    .otherwise("break")
    .alias("status"),
))

print(
    "Expected gap (not a break): identified-customer net vs merchandise net = anonymous sales still in the KPIs."
)
display(spark.sql(
    """
    SELECT
      (SELECT SUM(line_revenue) FROM trade) AS merchandise_net,
      (SELECT SUM(line_revenue) FROM trade WHERE `Customer ID` IS NOT NULL) AS identified_customer_net,
      (SELECT SUM(line_revenue) FROM trade WHERE `Customer ID` IS NULL) AS anonymous_net,
      (SELECT SUM(net_revenue) FROM dashboard_top_customers) AS top10_customers_net
    """
))
