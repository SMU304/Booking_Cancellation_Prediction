# Databricks notebook source
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from datetime import date
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.feature_selection import f_classif
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from numpy import *
import calendar 
import pickle

# COMMAND ----------

# MAGIC %scala
# MAGIC val Booking_Hist = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_bookinginsight/final/v2/tF_Booking_Final_Hist/")
# MAGIC .createOrReplaceTempView("Booking_Hist")
# MAGIC 
# MAGIC val Booking = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_bookinginsight/final/v2/tF_Booking_Final/")
# MAGIC .createOrReplaceTempView("Booking")
# MAGIC 
# MAGIC val Cust = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_bookinginsight/final/v2/tD_Cust")
# MAGIC .createOrReplaceTempView("Cust")
# MAGIC 
# MAGIC val Loc =
# MAGIC spark.read.load("/mnt/commercialbdi_pub_bookinginsight/final/v2/tD_Location")
# MAGIC .select(("Site_Cd"),("Site_Dsc"),("Site"),("SiteType_Cd"),("City_Cd"),("City_Dsc"),("City"),("UN_Loc_City_Cd"),("Country_Cd"),("Country_Dsc"),("SubArea_Cd"),("SubArea_Dsc"),("Area_Cd"),("Area_Dsc"),("Region_Cd"),("Region_Dsc"))
# MAGIC .distinct
# MAGIC .createOrReplaceTempView("Loc")
# MAGIC 
# MAGIC val booking_concern =
# MAGIC spark.read.load("/mnt/commercialbdi_pub_bookinginsight/final/v2/tD_Cust")
# MAGIC .select(("Cust_Cd"),("CustConcern_Cd"),("Concern_Dsc"),("Cust_Brand_Cd"))
# MAGIC .distinct
# MAGIC .createOrReplaceTempView("booking_concern")
# MAGIC 
# MAGIC val prod =
# MAGIC spark.read.load("/mnt/commercialbdi_pub_bookinginsight/final/v2/td_contractprodsegment_master")
# MAGIC .select(("ContractProdSegment_Master_Key"),("Contract_Product_Lvl1"),("Contract_Product_Segment"))
# MAGIC .distinct
# MAGIC .createOrReplaceTempView("prod")
# MAGIC 
# MAGIC val hierarchy =
# MAGIC spark.read.load("/mnt/commercialbdi_pub_bookinginsight/final/v2/td_contractprodhierarchy_master")
# MAGIC .select(("ContractProdHierarchyMaster_Key"),("Contract_Product"),("Contract_Product_Lvl2"))
# MAGIC .distinct
# MAGIC .createOrReplaceTempView("hierarchy")

# COMMAND ----------

# MAGIC %python
# MAGIC  
# MAGIC df_Unique= spark.sql("""
# MAGIC select
# MAGIC Distinct
# MAGIC b.Container_Type,
# MAGIC b.Lopfi,
# MAGIC b.Dipla,
# MAGIC b.Place_of_Receipt,
# MAGIC b.Product_Delivery,
# MAGIC b.TRUE_POD_SITE_CD,
# MAGIC cust.CustValProp_Dsc,
# MAGIC cust.CustConcern_Cd,
# MAGIC cust.Vertical,
# MAGIC b.Contract_Product_Lvl1,
# MAGIC pd.Contract_Product_Segment,
# MAGIC hy.Contract_Product,
# MAGIC b.Contract_Allocation_Type_Code,
# MAGIC bcn.CustConcern_Cd as Booking_Concern_Cd,
# MAGIC b.Is_First_Vessel
# MAGIC from booking as b
# MAGIC  
# MAGIC left join cust as cust
# MAGIC on cust.Cust_Brand_Cd = b.Cust_Brand_Cd
# MAGIC  
# MAGIC left join booking_concern as bcn
# MAGIC on bcn.Cust_Brand_Cd = concat (b.Booked_By_Cust_CD,'|',b.Brand_cd)
# MAGIC  
# MAGIC left join Loc as l
# MAGIC on l.Site_Cd=b.Place_of_Receipt
# MAGIC  
# MAGIC left join prod as pd
# MAGIC on pd.ContractProdSegment_Master_Key=b.ContractProdSegment_Master_Key
# MAGIC  
# MAGIC left join hierarchy as hy
# MAGIC on hy.ContractProdHierarchyMaster_Key=b.ContractProdHierarchyMaster_Key
# MAGIC   
# MAGIC where b.Booking_Date between '2004-01-01' and current_date() 
# MAGIC and b.Is_First_Vessel ='True' and b.Shipment_Status_Desc != 'Migrated Off'
# MAGIC  
# MAGIC  
# MAGIC union
# MAGIC  
# MAGIC  
# MAGIC select distinct
# MAGIC BH.Container_Type,
# MAGIC BH.Lopfi,
# MAGIC BH.Dipla,
# MAGIC BH.Place_of_Receipt,
# MAGIC BH.Product_Delivery,
# MAGIC BH.TRUE_POD_SITE_CD,
# MAGIC cust.CustValProp_Dsc,
# MAGIC cust.CustConcern_Cd,
# MAGIC cust.Vertical,
# MAGIC BH.Contract_Product_Lvl1,
# MAGIC pd.Contract_Product_Segment,
# MAGIC hy.Contract_Product,
# MAGIC BH.Contract_Allocation_Type_Code,
# MAGIC bcn.CustConcern_Cd as Booking_Concern_Cd,
# MAGIC BH.Is_First_Vessel
# MAGIC  
# MAGIC from Booking_Hist as BH
# MAGIC  
# MAGIC left join cust as cust
# MAGIC on cust.Cust_Brand_Cd = BH.Cust_Brand_Cd
# MAGIC  
# MAGIC left join booking_concern as bcn
# MAGIC on bcn.Cust_Brand_Cd = concat (BH.Booked_By_Cust_CD,'|',BH.Brand_cd)
# MAGIC  
# MAGIC left join Loc as l
# MAGIC on l.Site_Cd=BH.Place_of_Receipt
# MAGIC  
# MAGIC left join prod as pd
# MAGIC on pd.ContractProdSegment_Master_Key=BH.ContractProdSegment_Master_Key
# MAGIC  
# MAGIC left join hierarchy as hy
# MAGIC on hy.ContractProdHierarchyMaster_Key=BH.ContractProdHierarchyMaster_Key
# MAGIC  
# MAGIC where BH.Booking_Date between '2004-01-01' and  current_date()  
# MAGIC and BH.Is_First_Vessel ='True' and BH.Shipment_Status_Desc != 'Migrated Off'
# MAGIC  
# MAGIC """)
# MAGIC  
# MAGIC temp_table_name = "df_Unique"
# MAGIC df_Unique.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

df_Unique = df_Unique.select("*").toPandas()

# COMMAND ----------

# DBTITLE 1,Joining the Tables 
# MAGIC %python
# MAGIC  
# MAGIC df= spark.sql("""
# MAGIC select
# MAGIC Distinct
# MAGIC b.SHIPMENT_NO,
# MAGIC b.Shipment_Status_Desc,
# MAGIC b.Container_no,
# MAGIC b.Container_units,
# MAGIC b.Cargo_Type,
# MAGIC b.Container_Size,
# MAGIC b.Container_Type,
# MAGIC b.Container_Height,
# MAGIC b.TEU,
# MAGIC b.Route_Cd,
# MAGIC b.Is_Hazardous,
# MAGIC b.Is_OOG,
# MAGIC b.Lopfi,
# MAGIC b.Dipla,
# MAGIC b.Place_of_Receipt,
# MAGIC b.Booking_Date,
# MAGIC b.Product_Delivery,
# MAGIC b.TRUE_POD_SITE_CD,
# MAGIC b.LOPFI_ETA_DATE,
# MAGIC b.LIVE_REEFER_YN,
# MAGIC b.Brand_cd,
# MAGIC b.Actual_Departure_TS,
# MAGIC b.Exp_Departure_Date,
# MAGIC cust.CustValProp_Dsc,
# MAGIC cust.CustConcern_Cd,
# MAGIC cust.Vertical,
# MAGIC b.Contract_Product_Lvl1,
# MAGIC pd.Contract_Product_Segment,
# MAGIC hy.Contract_Product,
# MAGIC b.Contract_Allocation_Type_Code,
# MAGIC bcn.CustConcern_Cd as Booking_Concern_Cd
# MAGIC  
# MAGIC from booking as b
# MAGIC  
# MAGIC left join cust as cust
# MAGIC on cust.Cust_Brand_Cd = b.Cust_Brand_Cd
# MAGIC  
# MAGIC left join booking_concern as bcn
# MAGIC on bcn.Cust_Brand_Cd = concat (b.Booked_By_Cust_CD,'|',b.Brand_cd)
# MAGIC  
# MAGIC left join Loc as l
# MAGIC on l.Site_Cd=b.Place_of_Receipt
# MAGIC  
# MAGIC left join prod as pd
# MAGIC on pd.ContractProdSegment_Master_Key=b.ContractProdSegment_Master_Key
# MAGIC  
# MAGIC left join hierarchy as hy
# MAGIC on hy.ContractProdHierarchyMaster_Key=b.ContractProdHierarchyMaster_Key
# MAGIC    
# MAGIC where b.Booking_Date between '2017-01-01' and current_date() 
# MAGIC and b.Is_First_Vessel ='True' and b.Shipment_Status_Desc != 'Migrated Off'
# MAGIC  
# MAGIC  
# MAGIC union
# MAGIC  
# MAGIC  
# MAGIC select distinct
# MAGIC BH.SHIPMENT_NO,
# MAGIC BH.Shipment_Status_Desc,
# MAGIC BH.Container_no,
# MAGIC BH.Container_units,
# MAGIC BH.Cargo_Type,
# MAGIC BH.Container_Size,
# MAGIC BH.Container_Type,
# MAGIC BH.Container_Height,
# MAGIC BH.TEU,
# MAGIC BH.Route_Cd,
# MAGIC BH.Is_Hazardous,
# MAGIC BH.Is_OOG,
# MAGIC BH.Lopfi,
# MAGIC BH.Dipla,
# MAGIC BH.Place_of_Receipt,
# MAGIC BH.Booking_Date,
# MAGIC BH.Product_Delivery,
# MAGIC BH.TRUE_POD_SITE_CD,
# MAGIC BH.LOPFI_ETA_DATE,
# MAGIC BH.LIVE_REEFER_YN,
# MAGIC BH.Brand_cd,
# MAGIC BH.Actual_Departure_TS,
# MAGIC BH.Exp_Departure_Date,
# MAGIC cust.CustValProp_Dsc,
# MAGIC cust.CustConcern_Cd,
# MAGIC cust.Vertical,
# MAGIC BH.Contract_Product_Lvl1,
# MAGIC pd.Contract_Product_Segment,
# MAGIC hy.Contract_Product,
# MAGIC BH.Contract_Allocation_Type_Code,
# MAGIC bcn.CustConcern_Cd as Booking_Concern_Cd
# MAGIC  
# MAGIC from Booking_Hist as BH
# MAGIC  
# MAGIC left join cust as cust
# MAGIC on cust.Cust_Brand_Cd = BH.Cust_Brand_Cd
# MAGIC  
# MAGIC left join booking_concern as bcn
# MAGIC on bcn.Cust_Brand_Cd = concat (BH.Booked_By_Cust_CD,'|',BH.Brand_cd)
# MAGIC  
# MAGIC left join Loc as l
# MAGIC on l.Site_Cd=BH.Place_of_Receipt
# MAGIC  
# MAGIC left join prod as pd
# MAGIC on pd.ContractProdSegment_Master_Key=BH.ContractProdSegment_Master_Key
# MAGIC  
# MAGIC left join hierarchy as hy
# MAGIC on hy.ContractProdHierarchyMaster_Key=BH.ContractProdHierarchyMaster_Key
# MAGIC  
# MAGIC where BH.Booking_Date between '2017-01-01' and  current_date() 
# MAGIC and BH.Is_First_Vessel ='True' and BH.Shipment_Status_Desc != 'Migrated Off'
# MAGIC  
# MAGIC """)
# MAGIC  
# MAGIC temp_table_name = "df"
# MAGIC df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# DBTITLE 1,Converting into Pandas DataFrame for further use!
df1 = df.select("*").toPandas()

# COMMAND ----------

# DBTITLE 1,Imputation Process
df1['Booking_Concern_Cd'] = df1['Booking_Concern_Cd'].fillna('-1')
df1['CustConcern_Cd'] = df1['CustConcern_Cd'].fillna('-1')
df1['CustValProp_Dsc'] = df1['CustValProp_Dsc'].fillna('-1')
df1['Vertical'] = df1['Vertical'].fillna('-1')
df1['Place_of_Receipt']=df1['Place_of_Receipt'].fillna('-1')
df1['TRUE_POD_SITE_CD']=df1['TRUE_POD_SITE_CD'].fillna('-1')

# COMMAND ----------

df_Unique['Booking_Concern_Cd'] = df_Unique['Booking_Concern_Cd'].fillna('-1')
df_Unique['CustConcern_Cd'] = df_Unique['CustConcern_Cd'].fillna('-1')
df_Unique['CustValProp_Dsc'] = df_Unique['CustValProp_Dsc'].fillna('-1')
df_Unique['Vertical'] = df_Unique['Vertical'].fillna('-1')
df_Unique['Place_of_Receipt']=df_Unique['Place_of_Receipt'].fillna('-1')
df_Unique['TRUE_POD_SITE_CD']=df_Unique['TRUE_POD_SITE_CD'].fillna('-1')

# COMMAND ----------

# DBTITLE 1,Assigning values of each Variables
Lopfi=pd.DataFrame(
{
  "Lopfi" : df_Unique["Lopfi"].unique().tolist(),
  "Encoded_Lopfi" : list(np.arange(0,len(df_Unique["Lopfi"].unique().tolist()))+1)
  })

Dipla=pd.DataFrame(
{
  "Dipla" : df_Unique["Dipla"].unique().tolist(),
  "Encoded_Dipla" : list(np.arange(0,len(df_Unique["Dipla"].unique().tolist())) +1)
  })

Place_of_Receipt=pd.DataFrame(
{
  "Place_of_Receipt" : df_Unique["Place_of_Receipt"].unique().tolist(),
  "Encoded_Place_of_Receipt" : list(np.arange(0,len(df_Unique["Place_of_Receipt"].unique().tolist()))+1)
  })

TRUE_POD_SITE_CD=pd.DataFrame(
{
  "TRUE_POD_SITE_CD" : df_Unique["TRUE_POD_SITE_CD"].unique().tolist(),
  "Encoded_TRUE_POD_SITE_CD" : list(np.arange(0,len(df_Unique["TRUE_POD_SITE_CD"].unique().tolist()))+1)
  })

CustConcern_Cd=pd.DataFrame(
{
  "CustConcern_Cd" : df_Unique["CustConcern_Cd"].unique().tolist(),
  "Encoded_CustConcern_Cd" : list(np.arange(0,len(df_Unique["CustConcern_Cd"].unique().tolist()))+1)
  })

Booking_Concern_Cd=pd.DataFrame(
{
  "Booking_Concern_Cd" : df_Unique["Booking_Concern_Cd"].unique().tolist(),
  "Encoded_Booking_Concern_Cd" : list(np.arange(0,len(df_Unique["Booking_Concern_Cd"].unique().tolist()))+1)
  })

Product_Delivery=pd.DataFrame(
{
  "Product_Delivery" : df_Unique["Product_Delivery"].unique().tolist(),
  "Encoded_Product_Delivery" : list(np.arange(0,len(df_Unique["Product_Delivery"].unique().tolist()))+1)
  })

CustValProp_Dsc=pd.DataFrame(
{
  "CustValProp_Dsc" : df_Unique["CustValProp_Dsc"].unique().tolist(),
  "Encoded_CustValProp_Dsc" : list(np.arange(0,len(df_Unique["CustValProp_Dsc"].unique().tolist()))+1)
  })

Vertical=pd.DataFrame(
{
  "Vertical" : df_Unique["Vertical"].unique().tolist(),
  "Encoded_Vertical" : list(np.arange(0,len(df_Unique["Vertical"].unique().tolist()))+1)
  })

Contract_Product_Lvl1=pd.DataFrame(
{
  "Contract_Product_Lvl1" : df_Unique["Contract_Product_Lvl1"].unique().tolist(),
  "Encoded_Contract_Product_Lvl1" : list(np.arange(0,len(df_Unique["Contract_Product_Lvl1"].unique().tolist()))+1)
  })

Contract_Product_Segment=pd.DataFrame(
{
  "Contract_Product_Segment" : df_Unique["Contract_Product_Segment"].unique().tolist(),
  "Encoded_Contract_Product_Segment" : list(np.arange(0,len(df_Unique["Contract_Product_Segment"].unique().tolist()))+1)
  })

Contract_Product=pd.DataFrame(
{
  "Contract_Product" : df_Unique["Contract_Product"].unique().tolist(),
  "Encoded_Contract_Product" : list(np.arange(0,len(df_Unique["Contract_Product"].unique().tolist()))+1)
  })

Contract_Allocation_Type_Code=pd.DataFrame(
{
  "Contract_Allocation_Type_Code" : df_Unique["Contract_Allocation_Type_Code"].unique().tolist(),
  "Encoded_Contract_Allocation_Type_Code" : list(np.arange(0,len(df_Unique["Contract_Allocation_Type_Code"].unique().tolist()))+1)
  })

Container_Type=pd.DataFrame(
{
  "Container_Type" : df_Unique["Container_Type"].unique().tolist(),
  "Encoded_Container_Type" : list(np.arange(0,len(df_Unique["Container_Type"].unique().tolist()))+1)
  }) 

# COMMAND ----------

# DBTITLE 1,Temp Table
temp1 = spark.createDataFrame(Lopfi)
temp_table_name = "Lf"
temp1.createOrReplaceTempView(temp_table_name)

temp2 = spark.createDataFrame(Dipla)
temp_table_name = "Dp"
temp2.createOrReplaceTempView(temp_table_name)

temp3 = spark.createDataFrame(Place_of_Receipt)
temp_table_name = "PoR"
temp3.createOrReplaceTempView(temp_table_name)

temp4 = spark.createDataFrame(TRUE_POD_SITE_CD)
temp_table_name = "POD"
temp4.createOrReplaceTempView(temp_table_name)

temp5 = spark.createDataFrame(CustConcern_Cd)
temp_table_name = "CustCd"
temp5.createOrReplaceTempView(temp_table_name)

temp6 = spark.createDataFrame(Booking_Concern_Cd)
temp_table_name = "BCd"
temp6.createOrReplaceTempView(temp_table_name)

temp7 = spark.createDataFrame(Product_Delivery)
temp_table_name = "PD"
temp7.createOrReplaceTempView(temp_table_name)

temp8 = spark.createDataFrame(CustValProp_Dsc)
temp_table_name = "CVP"
temp8.createOrReplaceTempView(temp_table_name)

temp9 = spark.createDataFrame(Vertical)
temp_table_name = "VT"
temp9.createOrReplaceTempView(temp_table_name)

temp10= spark.createDataFrame(Contract_Product_Lvl1)
temp_table_name = "CPL"
temp10.createOrReplaceTempView(temp_table_name)

temp11= spark.createDataFrame(Contract_Product_Segment)
temp_table_name = "CPS"
temp11.createOrReplaceTempView(temp_table_name)

temp12= spark.createDataFrame(Contract_Product)
temp_table_name = "CP"
temp12.createOrReplaceTempView(temp_table_name)

temp13= spark.createDataFrame(Contract_Allocation_Type_Code)
temp_table_name = "CAT"
temp13.createOrReplaceTempView(temp_table_name)

temp14= spark.createDataFrame(Container_Type)
temp_table_name = "CT"
temp14.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %scala
# MAGIC spark.sql(s"""
# MAGIC 
# MAGIC select
# MAGIC Lopfi,
# MAGIC Encoded_Lopfi 
# MAGIC from Lf
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Lopfi/")
# MAGIC 
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Dipla,
# MAGIC Encoded_Dipla 
# MAGIC from Dp
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Dipla/")
# MAGIC 
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Place_of_Receipt,
# MAGIC Encoded_Place_of_Receipt 
# MAGIC from PoR
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Place_of_Receipt/")
# MAGIC 
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC TRUE_POD_SITE_CD,
# MAGIC Encoded_TRUE_POD_SITE_CD
# MAGIC from POD
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/TRUE_POD_SITE_CD/")
# MAGIC 
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC CustConcern_Cd ,
# MAGIC Encoded_CustConcern_Cd
# MAGIC from CustCd
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/CustConcern_Cd/")
# MAGIC 
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Booking_Concern_Cd ,
# MAGIC Encoded_Booking_Concern_Cd
# MAGIC from BCd
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Booking_Concern_Cd/")
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Product_Delivery ,
# MAGIC Encoded_Product_Delivery
# MAGIC from PD
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Product_Delivery/")
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC CustValProp_Dsc ,
# MAGIC Encoded_CustValProp_Dsc
# MAGIC from CVP
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/CustValProp_Dsc/")
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Vertical ,
# MAGIC Encoded_Vertical
# MAGIC from VT
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Vertical/")

# COMMAND ----------

# MAGIC %scala
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Contract_Product_Lvl1 ,
# MAGIC Encoded_Contract_Product_Lvl1
# MAGIC from CPL
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Contract_Product_Lvl1/")
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Contract_Product_Segment ,
# MAGIC Encoded_Contract_Product_Segment
# MAGIC from CPS
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Contract_Product_Segment/")
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Contract_Product ,
# MAGIC Encoded_Contract_Product
# MAGIC from CP
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Contract_Product/")
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Contract_Allocation_Type_Code ,
# MAGIC Encoded_Contract_Allocation_Type_Code
# MAGIC from CAT
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Contract_Allocation_Type_Code/")
# MAGIC 
# MAGIC 
# MAGIC spark.sql(s"""
# MAGIC select
# MAGIC Container_Type ,
# MAGIC Encoded_Container_Type
# MAGIC from CT
# MAGIC """)
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .option("mergeSchema", "true")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Container_Type/")

# COMMAND ----------

# DBTITLE 1,Applying the Logic 
#for converting historical active status into complete status and also converting two different status into single called cancelled where ever the Actual departure #data is not null
df1.loc[(df1.Shipment_Status_Desc == 'Active') & (df1.Actual_Departure_TS.notnull()), 'Shipment_Status_Desc'] = 'Completed'
df1['Shipment_Status_Desc'] = df1['Shipment_Status_Desc'].replace(['Cancelled by Carrier','Cancelled by Customer'],'Cancelled')

# COMMAND ----------

df1=df1.drop(['Actual_Departure_TS'],axis=1)

# COMMAND ----------

df1=df1.drop_duplicates()

# COMMAND ----------

# DBTITLE 1,Removing Container No. column 
df1=df1.drop(['Container_no'],axis=1)

# COMMAND ----------

# DBTITLE 1,Aggregating the values(TEU & Container Units)
df_sum_TEU=df1.groupby(['SHIPMENT_NO'])[['TEU','Container_units']].sum()
df_sum_TEU=df_sum_TEU.reset_index()

# COMMAND ----------

# DBTITLE 1,Removing the Duplicates values in the df1 DataFrame by Shipment.No
df1=df1.drop_duplicates('SHIPMENT_NO')

# COMMAND ----------

df1=df1.drop(['Container_Size','Container_Height'],axis=1)

# COMMAND ----------

# DBTITLE 1,Merging two DataFrame df1 and df_sum_TEU by Shipment No.
df_Final=pd.merge(df1, df_sum_TEU, how='inner',on=["SHIPMENT_NO"])

# COMMAND ----------

# DBTITLE 1,Dropping the extra columns & renaming the new one as it is !
df_Final=df_Final.drop(['TEU_x','Container_units_x'],axis=1)
df_Final.rename(columns = {'TEU_y' : 'TEU','Container_units_y':'Container_units'}, inplace = True)

# COMMAND ----------

# DBTITLE 1,Slicing the dataset for result purpose(For report creation)
Final_Result_Train=df_Final.copy()

# COMMAND ----------

Is_Hazardous_dict = {True:1,False:2}
df_Final['Is_Hazardous'] =df_Final['Is_Hazardous'].map(Is_Hazardous_dict)

Is_OOG_dict = {True:1,False:2}
df_Final['Is_OOG'] =df_Final['Is_OOG'].map(Is_OOG_dict)

LIVE_REEFER_YN_dict = {True:1,False:2}
df_Final['LIVE_REEFER_YN'] =df_Final['LIVE_REEFER_YN'].map(LIVE_REEFER_YN_dict)  

Cargo_Type_dict = {'DRY':1,'REEF':2}
df_Final['Cargo_Type'] = df_Final['Cargo_Type'].map(Cargo_Type_dict)

# COMMAND ----------

Lopfi_dict=Lopfi.set_index("Lopfi").to_dict()["Encoded_Lopfi"]
Dipla_dict=Dipla.set_index("Dipla").to_dict()["Encoded_Dipla"]
Place_of_Receipt_dict=Place_of_Receipt.set_index("Place_of_Receipt").to_dict()["Encoded_Place_of_Receipt"]
TRUE_POD_SITE_CD_dict=TRUE_POD_SITE_CD.set_index("TRUE_POD_SITE_CD").to_dict()["Encoded_TRUE_POD_SITE_CD"]
CustConcern_Cd_dict=CustConcern_Cd.set_index("CustConcern_Cd").to_dict()["Encoded_CustConcern_Cd"]
Booking_Concern_Cd_dict=Booking_Concern_Cd.set_index("Booking_Concern_Cd").to_dict()["Encoded_Booking_Concern_Cd"]
Product_Delivery_dict=Product_Delivery.set_index("Product_Delivery").to_dict()["Encoded_Product_Delivery"]
CustValProp_Dsc_dict=CustValProp_Dsc.set_index("CustValProp_Dsc").to_dict()["Encoded_CustValProp_Dsc"]
Vertical_dict=Vertical.set_index("Vertical").to_dict()["Encoded_Vertical"]
Contract_Product_Lvl1_dict=Contract_Product_Lvl1.set_index("Contract_Product_Lvl1").to_dict()["Encoded_Contract_Product_Lvl1"]
Contract_Product_Segment_dict=Contract_Product_Segment.set_index("Contract_Product_Segment").to_dict()["Encoded_Contract_Product_Segment"]
Contract_Product_dict=Contract_Product.set_index("Contract_Product").to_dict()["Encoded_Contract_Product"]
Contract_Allocation_Type_Code_dict=Contract_Allocation_Type_Code.set_index("Contract_Allocation_Type_Code").to_dict()["Encoded_Contract_Allocation_Type_Code"]
Container_Type_dict=Container_Type.set_index("Container_Type").to_dict()["Encoded_Container_Type"]

# COMMAND ----------

df_Final['Lopfi'] =df_Final['Lopfi'].map(Lopfi_dict)
df_Final['Dipla'] =df_Final['Dipla'].map(Dipla_dict)
df_Final['Place_of_Receipt'] =df_Final['Place_of_Receipt'].map(Place_of_Receipt_dict)
df_Final['TRUE_POD_SITE_CD'] =df_Final['TRUE_POD_SITE_CD'].map(TRUE_POD_SITE_CD_dict)
df_Final['CustConcern_Cd'] =df_Final['CustConcern_Cd'].map(CustConcern_Cd_dict)
df_Final['Booking_Concern_Cd'] =df_Final['Booking_Concern_Cd'].map(Booking_Concern_Cd_dict)
df_Final['Product_Delivery'] =df_Final['Product_Delivery'].map(Product_Delivery_dict)
df_Final['CustValProp_Dsc'] =df_Final['CustValProp_Dsc'].map(CustValProp_Dsc_dict)
df_Final['Vertical'] =df_Final['Vertical'].map(Vertical_dict)
df_Final['Contract_Product_Lvl1'] =df_Final['Contract_Product_Lvl1'].map(Contract_Product_Lvl1_dict)
df_Final['Contract_Product_Segment'] =df_Final['Contract_Product_Segment'].map(Contract_Product_Segment_dict)
df_Final['Contract_Product'] =df_Final['Contract_Product'].map(Contract_Product_dict)
df_Final['Contract_Allocation_Type_Code'] =df_Final['Contract_Allocation_Type_Code'].map(Contract_Allocation_Type_Code_dict)
df_Final['Container_Type'] =df_Final['Container_Type'].map(Container_Type_dict)

# COMMAND ----------

# DBTITLE 1,Label Encoding shipment status seperately 
Shipment_Status_Desc_dict = {'Completed': 0, 'Cancelled': 1,'Active':2}
df_Final['Shipment_Status_Desc'] =df_Final['Shipment_Status_Desc'].map(Shipment_Status_Desc_dict)

# COMMAND ----------

# DBTITLE 1,Training dataset
train_data = df_Final.copy()

# COMMAND ----------

# DBTITLE 1,Removing the extra Active status records which not fits into the logic
train_data=train_data[train_data['Shipment_Status_Desc'] != 2]
train_data.drop(['LOPFI_ETA_DATE','Exp_Departure_Date','Route_Cd','Booking_Date'], axis = 1, inplace = True)

# COMMAND ----------

# DBTITLE 1,Spliting the data (X & Y)
X=train_data.drop(['Shipment_Status_Desc'],axis=1)
y=train_data[['Shipment_Status_Desc']]

# COMMAND ----------

# DBTITLE 1,Training data =80% & test data =20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# COMMAND ----------

Xtrain =X_train.drop(['SHIPMENT_NO'],axis=1)
Xtest =X_test.drop(['SHIPMENT_NO'],axis=1)

# COMMAND ----------

# DBTITLE 1,Training the model by (Random Forest classifier) Method
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=-1, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
rfc.fit(Xtrain, y_train)
y_pred =pd.DataFrame(rfc.predict(Xtest),columns = ['Prediction'])
from sklearn.metrics import accuracy_score
print('Model accuracy score with 10 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# COMMAND ----------

import pickle 

# COMMAND ----------

# DBTITLE 1,Pickle File 
pickle.dump(rfc, open('/dbfs/FileStore/rfc.pkl','wb'))

# COMMAND ----------

pickled_model = pickle.load(open('/dbfs/FileStore/rfc.pkl', 'rb'))

# COMMAND ----------

train_data_Prediction=pd.DataFrame(rfc.predict(Xtrain),columns = ['Prediction'])

# COMMAND ----------

trainX_y=pd.concat([Xtrain,y_train],axis=1)

# COMMAND ----------

train_data_Prediction['Probability_of_Cancellation_Percentage']=pd.DataFrame(rfc.predict_proba(Xtrain)[:,1])

# COMMAND ----------

train_results=pd.concat([trainX_y.reset_index(drop = True),train_data_Prediction],axis=1)

# COMMAND ----------

train_results_Final = pd.concat([X_train['SHIPMENT_NO'].reset_index(drop = True),train_results],axis = 1)

# COMMAND ----------

TestX_y=pd.concat([Xtest,y_test],axis=1)

# COMMAND ----------

y_pred['Probability_of_Cancellation_Percentage']=pd.DataFrame(rfc.predict_proba(Xtest)[:,1])

# COMMAND ----------

Test_results=pd.concat([TestX_y.reset_index(drop = True),y_pred],axis=1)

# COMMAND ----------

Test_results_Final = pd.concat([X_test['SHIPMENT_NO'].reset_index(drop = True),Test_results], axis = 1)

# COMMAND ----------

Historical_Output=pd.concat([train_results_Final.reset_index(drop=True),Test_results_Final],join="outer",axis=0,ignore_index=True)

# COMMAND ----------

Historical_Output["Probability_of_Cancellation_Percentage"]=Historical_Output["Probability_of_Cancellation_Percentage"].apply(lambda x: round(x,4)*100)

# COMMAND ----------

Final_Result_Train=Final_Result_Train[Final_Result_Train['Shipment_Status_Desc']!='Active']

# COMMAND ----------

Final_Result_Train=Final_Result_Train.reset_index()

# COMMAND ----------

Historical_Output = pd.merge(Final_Result_Train,Historical_Output[['SHIPMENT_NO','Prediction','Probability_of_Cancellation_Percentage']],on='SHIPMENT_NO', how='inner')

# COMMAND ----------

Historical_Output["Prediction"]=Historical_Output["Prediction"].replace(to_replace=[0,1],value=["Completed","Cancelled"])
Historical_Output["Shipment_Status_Desc"]=Historical_Output["Shipment_Status_Desc"].replace(to_replace=[0,1],value=["Completed","Cancelled"])

# COMMAND ----------

Historical_Output=Historical_Output.drop(['Prediction'],axis=1)

# COMMAND ----------

Historical_Output['Prediction'] = Historical_Output['Probability_of_Cancellation_Percentage'].map(lambda x: 1 if x>=38 else 0)

# COMMAND ----------

Historical_Output["Prediction"]=Historical_Output["Prediction"].replace(to_replace=[0,1],value=["Completed","Cancelled"])

# COMMAND ----------

Historical_Output.to_csv('/dbfs/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Historical_Output', sep=',',header=True, index=True)

# COMMAND ----------

# MAGIC %scala
# MAGIC val testdf = spark.read.format("com.databricks.spark.csv")
# MAGIC .option("header", "true")
# MAGIC .load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Historical_Output") 
# MAGIC .select(("Booking_Date"),("Cargo_Type"),("Container_Type"),("Route_Cd"),("Is_Hazardous"),("Is_OOG"),("Lopfi"),("Dipla"),("Place_of_Receipt"),("Product_Delivery"),("TRUE_POD_SITE_CD"),("LOPFI_ETA_DATE"),("LIVE_REEFER_YN"),("Brand_cd"),("Exp_Departure_Date"),("CustValProp_Dsc"),("CustConcern_Cd"),("Vertical"),("Contract_Product_Lvl1"),("Contract_Product_Segment"),("Contract_Product"),("Contract_Allocation_Type_Code"),("Booking_Concern_Cd"),("TEU"),("Container_units"),("SHIPMENT_NO"),("Shipment_Status_Desc"),("Probability_of_Cancellation_Percentage"),("Prediction")) 
# MAGIC // //testdf.show()
# MAGIC 
# MAGIC testdf
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/historical_output")

# COMMAND ----------

# MAGIC %scala
# MAGIC val Active_Hist = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/final_output/")
# MAGIC .createOrReplaceTempView("Active_Hist")

# COMMAND ----------

df_Active_History= spark.sql("""
select * from Active_Hist
""")

# COMMAND ----------

df_Active_History=df_Active_History.select("*").toPandas()

# COMMAND ----------

def comparision(Historical_Output,df_Active_History):
  df_Comparision=df_Active_History[~df_Active_History["SHIPMENT_NO"].isin(Historical_Output["SHIPMENT_NO"])]
  return(df_Comparision)

# COMMAND ----------

df_Active_History=comparision(Historical_Output,df_Active_History)

# COMMAND ----------

df_Active_History.to_csv('/dbfs/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/df_Active_History', sep=',',header=True, index=True)

# COMMAND ----------

# MAGIC %scala
# MAGIC val testdf = spark.read.format("com.databricks.spark.csv")
# MAGIC .option("header", "true")
# MAGIC .load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/df_Active_History") 
# MAGIC .select(("SHIPMENT_NO"),("Prediction"),("Probability_of_Cancellation_Percentage")) 
# MAGIC // //testdf.show()
# MAGIC 
# MAGIC testdf
# MAGIC .write
# MAGIC .format("delta")
# MAGIC .mode("overwrite")
# MAGIC .save("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/final_output")