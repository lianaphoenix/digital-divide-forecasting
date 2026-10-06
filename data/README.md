# Data sources

The analysis downloads World Bank World Development Indicators (WDI) data at runtime. No raw API download is committed.

| Measure | WDI indicator |
| --- | --- |
| Internet users | `IT.NET.USER.ZS` |
| Mobile subscriptions | `IT.CEL.SETS.P2` |
| Fixed broadband | `IT.NET.BBND.P2` |
| Electricity access | `EG.ELC.ACCS.ZS` |
| GDP per capita | `NY.GDP.PCAP.CD` |
| Secondary school enrollment | `SE.SEC.ENRR` |
| Urban population | `SP.URB.TOTL.IN.ZS` |

The pipeline excludes aggregate entities and keeps years in which every index component has at least 100 reporting countries. A compact processed panel from the published rerun is retained in [`../outputs/digital_divide_panel.csv`](../outputs/digital_divide_panel.csv).
