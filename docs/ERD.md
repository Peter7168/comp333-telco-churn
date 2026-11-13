# Entity-relationship diagram (Phase 2)

Phase 1 had the three source 1 tables. Phase 2 adds the five source 2 tables.
Every customer table is one-to-one on the customer ID. Location to population
is many-to-one on zip code (1,626 zip codes used by 7,043 customers, out of
1,671 zip codes in the population file).

```mermaid
erDiagram
    CUSTOMER_PROFILES ||--|| SERVICE_SUBSCRIPTIONS : "customerID (1:1)"
    CUSTOMER_PROFILES ||--|| ACCOUNT_BILLING : "customerID (1:1)"
    CUSTOMER_PROFILES ||--|| DEMOGRAPHICS : "customerID = Customer ID (1:1)"
    CUSTOMER_PROFILES ||--|| SERVICES : "customerID = Customer ID (1:1)"
    CUSTOMER_PROFILES ||--|| STATUS : "customerID = Customer ID (1:1)"
    CUSTOMER_PROFILES ||--|| LOCATION : "customerID = Customer ID (1:1)"
    LOCATION }o--|| POPULATION : "Zip Code (N:1)"

    CUSTOMER_PROFILES {
        string customerID PK
        string gender
        int SeniorCitizen
        string Partner
        string Dependents
        int tenure
    }
    SERVICE_SUBSCRIPTIONS {
        string customerID PK,FK
        string PhoneService
        string MultipleLines
        string InternetService
        string OnlineSecurity
        string OnlineBackup
        string DeviceProtection
        string TechSupport
        string StreamingTV
        string StreamingMovies
    }
    ACCOUNT_BILLING {
        string customerID PK,FK
        string Contract
        string PaperlessBilling
        string PaymentMethod
        float MonthlyCharges
        string TotalCharges
        string Churn
    }
    DEMOGRAPHICS {
        string CustomerID PK,FK
        int Age
        string Under30
        string SeniorCitizen
        string Married
        string Dependents
        int NumberOfDependents
    }
    SERVICES {
        string CustomerID PK,FK
        int NumberOfReferrals
        int TenureInMonths
        string Offer
        string InternetType
        int AvgMonthlyGBDownload
        float AvgMonthlyLongDistanceCharges
        string StreamingMusic
        string UnlimitedData
        string Contract
        string PaymentMethod
        float TotalCharges
        float TotalRefunds
        int TotalExtraDataCharges
        float TotalLongDistanceCharges
        float TotalRevenue
    }
    STATUS {
        string CustomerID PK,FK
        int SatisfactionScore "leaky, excluded"
        string CustomerStatus "leaky, excluded"
        string ChurnLabel
        int ChurnScore "leaky, excluded"
        int CLTV "excluded"
        string ChurnCategory "leaky, excluded"
        string ChurnReason "leaky, excluded"
    }
    LOCATION {
        string CustomerID PK,FK
        string City
        int ZipCode FK
        float Latitude
        float Longitude
    }
    POPULATION {
        int ZipCode PK
        int Population
    }
```

Source 2 tables list only the main columns. The full column-by-column mapping
is in `reports/tables/schema_mapping.csv`.
