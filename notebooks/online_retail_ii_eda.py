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
# MAGIC Raw file lives on a **Volume**. The notebook reads a **table**.
# MAGIC
# MAGIC - Volume file: `/Volumes/workspace/default/my_files/online_retail/online_retail_ii.parquet`  
# MAGIC   (same data as `online_retail_II.xlsx`, both year tabs, plus `source_sheet`)
# MAGIC - Table: `workspace.default.online_retail_ii` in the existing `workspace.default` schema — we do not create a new schema.
# MAGIC
# MAGIC `Customer ID` has a space, so the table is created with Delta column mapping. If the table already exists, the create step is skipped.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

dbutils.widgets.text("table_name", "workspace.default.online_retail_ii")
dbutils.widgets.text(
    "volume_parquet",
    "/Volumes/workspace/default/my_files/online_retail/online_retail_ii.parquet",
)

TABLE = dbutils.widgets.get("table_name")
VOLUME_PARQUET = dbutils.widgets.get("volume_parquet")

if not spark.catalog.tableExists(TABLE):
    # Column mapping: Delta otherwise rejects the space in `Customer ID`.
    spark.sql(
        f"""
        CREATE TABLE {TABLE}
        TBLPROPERTIES (
          'delta.minReaderVersion' = '2',
          'delta.minWriterVersion' = '5',
          'delta.columnMapping.mode' = 'name'
        )
        AS
        SELECT * FROM read_files('{VOLUME_PARQUET}', format => 'parquet')
        """
    )
    print("created", TABLE, "from", VOLUME_PARQUET)
else:
    print("using existing", TABLE)

raw = spark.table(TABLE)
raw.createOrReplaceTempView("retail_raw")
print("columns:", raw.columns)
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

PRICE_COL = "Price" if "Price" in raw.columns else "UnitPrice"
assert PRICE_COL in raw.columns, raw.columns

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

w = Window.orderBy(F.desc("net_revenue"))
products = products.withColumn("rank", F.row_number().over(w)).withColumn(
    "cum_share", F.sum("net_revenue").over(w) / F.sum("net_revenue").over(Window.partitionBy())
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
cw = Window.orderBy(F.desc("net_revenue"))
customers = customers.withColumn("rank", F.row_number().over(cw)).withColumn(
    "cum_share", F.sum("net_revenue").over(cw) / F.sum("net_revenue").over(Window.partitionBy())
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
geo = geo.withColumn("share", F.col("net_revenue") / F.sum("net_revenue").over(Window.partitionBy()))
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
