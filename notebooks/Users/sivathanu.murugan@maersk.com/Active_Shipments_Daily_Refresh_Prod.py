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

# DBTITLE 1,Joining the Tables 
# MAGIC %python
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
# MAGIC where b.Booking_Date between '2022-01-01' and current_date()
# MAGIC and b.Is_First_Vessel ='True' and b.Shipment_Status_Desc != 'Migrated Off' 
# MAGIC  
# MAGIC """)
# MAGIC  
# MAGIC temp_table_name = "df"
# MAGIC df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# DBTITLE 1,Converting into Pandas DataFrame for further use!
df1 = df.select("*").toPandas()

# COMMAND ----------

# DBTITLE 1,Imputation Method
df1['Booking_Concern_Cd'] = df1['Booking_Concern_Cd'].fillna('-1')
df1['CustConcern_Cd'] = df1['CustConcern_Cd'].fillna('-1')
df1['CustValProp_Dsc'] = df1['CustValProp_Dsc'].fillna('-1')
df1['Vertical'] = df1['Vertical'].fillna('-1')
df1['Place_of_Receipt']=df1['Place_of_Receipt'].fillna('-1')
df1['TRUE_POD_SITE_CD']=df1['TRUE_POD_SITE_CD'].fillna('-1')

# COMMAND ----------

# DBTITLE 1,Applying the Logic 
#for converting historical active status into complete status and also converting two different status into single called cancelled where ever the Actual departure #data is not null
df1.loc[(df1.Shipment_Status_Desc == 'Active') & (df1.Actual_Departure_TS.notnull()), 'Shipment_Status_Desc'] = 'Completed'
df1['Shipment_Status_Desc'] = df1['Shipment_Status_Desc'].replace(['Cancelled by Carrier','Cancelled by Customer'],'Cancelled')

# COMMAND ----------

df1=df1[df1['Shipment_Status_Desc']=='Active']

# COMMAND ----------

# DBTITLE 1,Defining the Active Function
def new_active(df_old,df_new):
  df_new_active=df_new[~df_new["SHIPMENT_NO"].isin(df_old["SHIPMENT_NO"])]
  return(df_new_active)

# COMMAND ----------

df_active=df1[df1['Shipment_Status_Desc']=='Active']

# COMMAND ----------

df_active.shape

# COMMAND ----------

# MAGIC %scala
# MAGIC val df_Active_Hist = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/final_output/")
# MAGIC .createOrReplaceTempView("df_Active_Hist")

# COMMAND ----------

df_Active_His= spark.sql("""
select * from df_Active_Hist
""")

# COMMAND ----------

df_Active_History=df_Active_His.select("*").toPandas()

# COMMAND ----------

df_Active_History['Probability_of_Cancellation_Percentage'] = pd.to_numeric(df_Active_History['Probability_of_Cancellation_Percentage'])

# COMMAND ----------

df_Active_History.shape

# COMMAND ----------

df_New_Active=new_active(df_Active_History,df_active)

# COMMAND ----------

df_New_Active.shape

# COMMAND ----------

df_New_Active=df_New_Active.drop(['Actual_Departure_TS','Container_no','Container_Height','Container_Size'],axis=1)

# COMMAND ----------

df1=df1.drop(['Actual_Departure_TS'],axis=1)

# COMMAND ----------

df1=df1.drop_duplicates()

# COMMAND ----------

# DBTITLE 1,Removing Container No. column 
df1=df1.drop(['Container_no'],axis=1)

# COMMAND ----------

# DBTITLE 1,Aggregating the values(TEU & Container Units)
df_New_Active_sum_TEU=df_New_Active.groupby(['SHIPMENT_NO'])[['TEU','Container_units']].sum()
df_New_Active_sum_TEU=df_New_Active_sum_TEU.reset_index()

# COMMAND ----------

# DBTITLE 1,Removing the Duplicates values in the df1 DataFrame by Shipment.No
df1=df1.drop_duplicates('SHIPMENT_NO')

# COMMAND ----------

df1=df1.drop(['Container_Size','Container_Height'],axis=1)

# COMMAND ----------

# DBTITLE 1,Merging two DataFrame df1 and df_sum_TEU by Shipment No.
df_Final=pd.merge(df1, df_New_Active_sum_TEU, how='inner',on=["SHIPMENT_NO"])

# COMMAND ----------

# DBTITLE 1,Dropping the extra columns & renaming the new one as it is !
df_Final=df_Final.drop(['TEU_x','Container_units_x'],axis=1)
df_Final.rename(columns = {'TEU_y' : 'TEU','Container_units_y':'Container_units'}, inplace = True)

# COMMAND ----------

# DBTITLE 1,Slicing the dataset for result purpose(For report creation)
Final_Result_Testing=df_Final.copy()

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

# MAGIC %scala
# MAGIC val Lf = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Lopfi/")
# MAGIC .createOrReplaceTempView("Lf")
# MAGIC 
# MAGIC val Dp = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Dipla/")
# MAGIC .createOrReplaceTempView("Dp")
# MAGIC 
# MAGIC val PoR = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Place_of_Receipt/")
# MAGIC .createOrReplaceTempView("PoR")
# MAGIC 
# MAGIC val POD = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/TRUE_POD_SITE_CD/")
# MAGIC .createOrReplaceTempView("POD")
# MAGIC 
# MAGIC val CustCd = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/CustConcern_Cd/")
# MAGIC .createOrReplaceTempView("CustCd")
# MAGIC 
# MAGIC val BCd = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Booking_Concern_Cd/")
# MAGIC .createOrReplaceTempView("BCd")
# MAGIC 
# MAGIC val PD = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Product_Delivery/")
# MAGIC .createOrReplaceTempView("PD")
# MAGIC 
# MAGIC val CVP = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/CustValProp_Dsc/")
# MAGIC .createOrReplaceTempView("CVP")
# MAGIC 
# MAGIC val VT = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Vertical/")
# MAGIC .createOrReplaceTempView("VT")
# MAGIC 
# MAGIC val CPL = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Contract_Product_Lvl1/")
# MAGIC .createOrReplaceTempView("CPL")
# MAGIC 
# MAGIC val CPS = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Contract_Product_Segment/")
# MAGIC .createOrReplaceTempView("CPS")
# MAGIC 
# MAGIC val CP = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Contract_Product/")
# MAGIC .createOrReplaceTempView("CP")
# MAGIC 
# MAGIC val CAT = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Contract_Allocation_Type_Code/")
# MAGIC .createOrReplaceTempView("CAT")
# MAGIC 
# MAGIC val CT = 
# MAGIC spark.read.load("/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/Container_Type/")
# MAGIC .createOrReplaceTempView("CT")

# COMMAND ----------

Lopfi= spark.sql("""
select * from Lf
""")

Dipla= spark.sql("""
select * from Dp
""")

Place_of_Receipt=spark.sql("""
select * from PoR
""")

TRUE_POD_SITE_CD= spark.sql("""
select * from POD
""")

CustConcern_Cd= spark.sql("""
select * from CustCd
""")

Booking_Concern_Cd= spark.sql("""
select * from BCd
""")

Product_Delivery= spark.sql("""
select * from PD
""")

CustValProp_Dsc= spark.sql("""
select * from CVP
""")

Vertical= spark.sql("""
select * from VT
""")

Contract_Product_Lvl1= spark.sql("""
select * from CPL
""")

Contract_Product_Segment= spark.sql("""
select * from CPS
""")

Contract_Product= spark.sql("""
select * from CP
""")

Contract_Allocation_Type_Code= spark.sql("""
select * from CAT
""")

Container_Type= spark.sql("""
select * from CT
""")

# COMMAND ----------

Lopfi=Lopfi.select("*").toPandas()
Dipla=Dipla.select("*").toPandas()
Place_of_Receipt=Place_of_Receipt.select("*").toPandas()
TRUE_POD_SITE_CD=TRUE_POD_SITE_CD.select("*").toPandas()
CustConcern_Cd=CustConcern_Cd.select("*").toPandas()
Booking_Concern_Cd=Booking_Concern_Cd.select("*").toPandas()
Product_Delivery=Product_Delivery.select("*").toPandas()
CustValProp_Dsc=CustValProp_Dsc.select("*").toPandas()
Vertical=Vertical.select("*").toPandas()
Contract_Product_Lvl1=Contract_Product_Lvl1.select("*").toPandas()
Contract_Product_Segment=Contract_Product_Segment.select("*").toPandas()
Contract_Product=Contract_Product.select("*").toPandas()
Contract_Allocation_Type_Code=Contract_Allocation_Type_Code.select("*").toPandas()
Container_Type=Container_Type.select("*").toPandas()

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

df_Final['Lopfi']=df_Final['Lopfi'].map(Lopfi_dict)
df_Final['Dipla']=df_Final['Dipla'].map(Dipla_dict)
df_Final['Place_of_Receipt']=df_Final['Place_of_Receipt'].map(Place_of_Receipt_dict)
df_Final['TRUE_POD_SITE_CD']=df_Final['TRUE_POD_SITE_CD'].map(TRUE_POD_SITE_CD_dict)
df_Final['CustConcern_Cd']=df_Final['CustConcern_Cd'].map(CustConcern_Cd_dict)
df_Final['Booking_Concern_Cd']=df_Final['Booking_Concern_Cd'].map(Booking_Concern_Cd_dict)
df_Final['Product_Delivery']=df_Final['Product_Delivery'].map(Product_Delivery_dict)
df_Final['CustValProp_Dsc']=df_Final['CustValProp_Dsc'].map(CustValProp_Dsc_dict)
df_Final['Vertical']=df_Final['Vertical'].map(Vertical_dict)
df_Final['Contract_Product_Lvl1']=df_Final['Contract_Product_Lvl1'].map(Contract_Product_Lvl1_dict)
df_Final['Contract_Product_Segment']=df_Final['Contract_Product_Segment'].map(Contract_Product_Segment_dict)
df_Final['Contract_Product']=df_Final['Contract_Product'].map(Contract_Product_dict)
df_Final['Contract_Allocation_Type_Code']=df_Final['Contract_Allocation_Type_Code'].map(Contract_Allocation_Type_Code_dict)
df_Final['Container_Type']=df_Final['Container_Type'].map(Container_Type_dict)

# COMMAND ----------

# DBTITLE 1,If any New entries arises It will be Encoded as Zero.
df_Final['Lopfi'] = df_Final['Lopfi'].replace(np.nan, 0)
df_Final['Dipla'] = df_Final['Dipla'].replace(np.nan, 0)
df_Final['Place_of_Receipt'] = df_Final['Place_of_Receipt'].replace(np.nan, 0)
df_Final['TRUE_POD_SITE_CD'] = df_Final['TRUE_POD_SITE_CD'].replace(np.nan, 0)
df_Final['CustConcern_Cd'] = df_Final['CustConcern_Cd'].replace(np.nan, 0)
df_Final['Booking_Concern_Cd'] = df_Final['Booking_Concern_Cd'].replace(np.nan, 0)
df_Final['Product_Delivery'] = df_Final['Product_Delivery'].replace(np.nan, 0)
df_Final['CustValProp_Dsc'] = df_Final['CustValProp_Dsc'].replace(np.nan, 0)
df_Final['Vertical'] = df_Final['Vertical'].replace(np.nan, 0)
df_Final['Contract_Product_Lvl1'] = df_Final['Contract_Product_Lvl1'].replace(np.nan, 0)
df_Final['Contract_Product_Segment'] = df_Final['Contract_Product_Segment'].replace(np.nan, 0)
df_Final['Contract_Product'] = df_Final['Contract_Product'].replace(np.nan, 0)
df_Final['Contract_Allocation_Type_Code'] = df_Final['Contract_Allocation_Type_Code'].replace(np.nan, 0)
df_Final['Container_Type'] = df_Final['Container_Type'].replace(np.nan, 0)

# COMMAND ----------

pickled_model = pickle.load(open('/dbfs/FileStore/rfc.pkl','rb'))

# COMMAND ----------

# DBTITLE 1,Live data Validation-2022
df_Final=df_Final.drop(['Route_Cd','LOPFI_ETA_DATE','Exp_Departure_Date','Booking_Date'],axis=1)

# COMMAND ----------

df_Final1=df_Final.drop(['Shipment_Status_Desc','SHIPMENT_NO'], axis = 1)  

# COMMAND ----------

# DBTITLE 1,Prediction  
prediction_active = pd.DataFrame(pickled_model.predict(df_Final1),columns = ['pred'])

# COMMAND ----------

result = pd.concat([df_Final[['SHIPMENT_NO','Shipment_Status_Desc']].reset_index(drop = True),prediction_active], axis = 1)

# COMMAND ----------

# DBTITLE 1,Probability Calculation for prediction
Prob_percentage = pd.DataFrame(pickled_model.predict_proba(df_Final1)[:,1],columns=['Probability_of_Cancellation_Percentage'])

# COMMAND ----------

Result_Prob_percentage = pd.concat([result.reset_index(drop = True),Prob_percentage], axis = 1)

# COMMAND ----------

Result_Prob_percentage["Probability_of_Cancellation_Percentage"]=Result_Prob_percentage["Probability_of_Cancellation_Percentage"].apply(lambda x: round(x,4)*100)

# COMMAND ----------

# DBTITLE 1,Making Prediction According to Threshold Value
Result_Prob_percentage['Prediction'] = Result_Prob_percentage['Probability_of_Cancellation_Percentage'].map(lambda x: 1 if x>=38 else 0)

# COMMAND ----------

result_final_1=Final_Result_Testing.drop(['Shipment_Status_Desc'],axis=1)

# COMMAND ----------

Live_Output=result_final_1.merge(Result_Prob_percentage,how='inner',on='SHIPMENT_NO')
Live_Output=Live_Output.drop(['pred'],axis=1)

# COMMAND ----------

Live_Output=Live_Output[["SHIPMENT_NO","Prediction","Probability_of_Cancellation_Percentage"]]

# COMMAND ----------

# DBTITLE 1,Creating Function & Appending Live_Active Output with History_Active table 
def history_active(df_Active_History,Live_Output):
   df_Active_History=pd.concat([df_Active_History,Live_Output]).drop_duplicates()
   return(df_Active_History)

# COMMAND ----------

df_Active_History=history_active(df_Active_History,Live_Output)
df_Active_History

# COMMAND ----------

df_Active_History["Prediction"]=df_Active_History["Prediction"].replace(to_replace=[0,1],value=["Completed","Cancelled"])

# COMMAND ----------

# DBTITLE 1,Pulling the columns needed for Final Result 
df_Active_History=df_Active_History[["SHIPMENT_NO","Prediction","Probability_of_Cancellation_Percentage"]]

# COMMAND ----------

Probability_for_cancelled=df_Active_History.groupby('Prediction').agg({'Probability_of_Cancellation_Percentage':['min','max']})
Probability_for_cancelled

# COMMAND ----------

# DBTITLE 1,Saving the Result as CSV File 
df_Active_History.to_csv('/dbfs/mnt/commercialbdi_pub_commercialbdi/ML_cancellation/df_Active_History', sep=',',header=True, index=True)

# COMMAND ----------

# DBTITLE 1,Saving the CSV file into a Delta Table 
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