# Data

No raw World Bank download is committed to this repository.

The analysis retrieves data from the World Bank World Development Indicators API at runtime. The indicator codes used by the final pipeline are:

- `IT.NET.USER.ZS` — Individuals using the Internet (% of population)
- `IT.CEL.SETS.P2` — Mobile cellular subscriptions (per 100 people)
- `IT.NET.BBND.P2` — Fixed broadband subscriptions (per 100 people)
- `EG.ELC.ACCS.ZS` — Access to electricity (% of population)
- `NY.GDP.PCAP.CD` — GDP per capita (current US$)
- `SE.SEC.ENRR` — School enrollment, secondary (% gross)
- `SP.URB.TOTL.IN.ZS` — Urban population (% of total population)

The pipeline also removes aggregate/non-country entities before constructing the index and applies the documented 100-reporting-country threshold.
