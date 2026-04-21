Tallinna Tööstushariduskeskus
Noorem Tarkvaraarendaja
Masinõppel põhinev finantsandmete analüüsi ja riskihalduse süsteem
Kirjeldus
Õpilane: Martin Sild
Rühm: TARpv23

Tallinn 2025
SISUKORD
1. TEOREETILINE OSA	5
1.1 Python	5
1.2 LightGBM	5
1.3 Polars	5
1.4 scikit-learn	6
1.5 Optuna	6
1.6 ONNX	6
2. PRAKTILINE OSA	6
2.1 Tehnoloogiad ja tööriistad	6
2.2 Projekti struktuur	7
2.3 Andmete töötlemine	9
2.3.1	Andmete kogumine	9
2.3.2	Andmete laadimine	9
2.3.3	Tunnusearvutus	10
2.3.4	Sildimine	11
2.3.5	Andmestiku jaotamine	11
2.4 Masinõppe mudelite treenimine	11
2.4.1	Mudelite arhitektuur	11
2.4.2	Treenimisprotsess	12
2.4.3	Tunnusetähtsus	13
2.5 Riskihalduse süsteem	13
2.5.1	CVaR mootor (Conditional Value at Risk)	13
2.5.2	Kelly kriteerium (positsioonide suurustamine)	14
2.5.3	Maha-tõmbe kontroller (Drawdown Controller)	14
2.5.4	Riskimooduli otsustusahel	15
2.6 Valideerimine ja optimeerimine	15
2.6.1	12-kontrollpunktiga valideerimiskonveier	15
2.6.2	Ülesobitamise tuvastamine	16
2.6.3	Optuna hüperparameetrite optimeerimine	16
2.6.4	Baasmudelite võrdlus	17
2.6.5	Signaalide genereerimine	17

 
SISSEJUHATUS
Finantsturud genereerivad iga päev väga suuri andmemahtusid ning nende käsitsi analüüsimine on keeruline ja aeganõudev. Masinõppe meetodid võimaldavad töödelda suuri andmekogumeid, leida mustreid ning teha prognoose turu võimalike hinnaliikumiste kohta. Sellised süsteemid aitavad analüütikutel ja arendajatel teha paremaid otsuseid ning hinnata võimalikke riske.
Käesoleva lõputöö eesmärk on luua masinõppel põhinev finantsandmete analüüsi ja riskihalduse süsteem. Süsteem analüüsib ajaloolisi turuandmeid ning kasutab masinõppe mudeleid, et hinnata hinnaliikumise võimalikku suunda. Lisaks keskendub süsteem riskihaldusele, mis aitab vähendada võimalikke kahjusid ning parandada otsuste kvaliteeti erinevates turutingimustes.
Projekti käigus luuakse süsteemi arhitektuur, mis koosneb mitmest komponendist: andmete kogumine ja töötlemine, tunnuste (features) arvutamine, masinõppe mudelite treenimine ning riskihalduse moodul. Riskihalduse osa võimaldab hinnata turuga seotud riske ja kontrollida võimalikke kahjusid, kasutades erinevaid meetodeid ja analüüsi.
Töö praktilises osas arendatakse prototüüp, mis demonstreerib, kuidas masinõppe meetodeid saab kasutada finantsandmete analüüsimiseks ja riskihalduse toetamiseks. Lõpptulemuseks on toimiv süsteem, mis näitab, kuidas andmepõhised lahendused võivad aidata paremini mõista turu käitumist ning toetada otsustusprotsessi.
 
TEOREETILINE OSA
Python
Python on üks populaarsemaid programmeerimiskeeli andmeteaduse ja masinõppe valdkonnas, mille populaarsus tuleneb lihtsast süntaksist, suurest raamatukogude valikust ning aktiivsest arendajate kogukonnast. Python võimaldab töödelda suuri andmemahte, teostada statistilist analüüsi ning arendada masinõppe mudeleid. Andmeteaduses kasutatakse sageli selliseid raamistikke nagu scikit-learn, LightGBM ja Optuna. Finantsandmete analüüsis kasutatakse Pythonit tehniliste indikaatorite arvutamiseks, andmete töötlemiseks ning masinõppe mudelite treenimiseks, et prognoosida võimalikke hinnaliikumisi.
LightGBM
LightGBM (Light Gradient Boosting Machine) on Microsofti poolt arendatud masinõppe raamistik, mis põhineb gradient boosting algoritmil ja on optimeeritud suurte andmemahtude kiireks töötlemiseks. Selle peamised eelised on suur treeningkiirus, madal mälukasutus ning kõrge täpsus. LightGBM kasutab lehepõhist puu kasvatamise algoritmi (leaf-wise tree growth), mis võimaldab luua täpsemaid mudeleid väiksema arvutusressursiga. Finantsandmete analüüsis kasutatakse LightGBM-i hinnaliikumise prognoosimiseks, kuna see suudab töödelda suurt hulka tunnuseid ja leida keerukaid mustreid andmetes.
Polars
Polars on kaasaegne ja suure jõudlusega andmetöötluse raamistik, mis on mõeldud suurte andmekogumite töötlemiseks. See on kirjutatud Rust programmeerimiskeeles ning pakub sageli paremat jõudlust võrreldes traditsiooniliste andmetöötlusraamistikega. Polars kasutab veerupõhist (columnar) andmestruktuuri ja toetab paralleelset andmetöötlust, mis võimaldab suuri andmemahte töödelda väga kiiresti. Finantsandmete analüüsi süsteemis kasutatakse Polarsit andmete puhastamiseks, transformeerimiseks ja tunnuste arvutamiseks enne masinõppe mudelite treenimist.
 
scikit-learn
scikit-learn on üks tuntumaid Python'i masinõppe raamistikke, mis pakub laia valikut algoritme klassifitseerimiseks, regressiooniks ja klasterdamiseks. Lisaks sisaldab see mitmeid tööriistu andmete ettevalmistamiseks, mudelite hindamiseks ja valideerimiseks, näiteks ristvalideerimist (cross-validation), andmete normaliseerimist ja erinevaid hindamismeetrikaid. Käesolevas projektis kasutatakse scikit-learn raamatukogu andmete eelprotsessimiseks, mudelite valideerimiseks ning erinevate masinõppe meetodite võrdlemiseks.
Optuna
Optuna on hüperparameetrite optimeerimise raamistik, mis võimaldab automaatselt leida masinõppe mudelite jaoks parimaid seadistusi. Masinõppe mudelite jõudlus sõltub sageli hüperparameetritest, nagu õppimiskiirus, puude arv või maksimaalne sügavus. Optuna kasutab efektiivseid otsingualgoritme, et leida parim parameetrite kombinatsioon võimalikult väikese arvutusressursiga. Käesolevas projektis kasutatakse Optuna raamistikku LightGBM mudelite optimeerimiseks, et parandada prognoositäpsust.
ONNX
ONNX (Open Neural Network Exchange) on avatud standard masinõppe mudelite salvestamiseks ja jagamiseks erinevate platvormide vahel. See võimaldab treenitud mudeleid eksportida ja kasutada erinevates süsteemides ilma sõltuvuseta algsest arendusraamistikust, mis lihtsustab mudelite kasutamist tootmiskeskkonnas. Käesolevas projektis kasutatakse ONNX formaati masinõppe mudelite eksportimiseks, et võimaldada nende kiiret ja efektiivset kasutamist reaalajas prognoosimisel.
 
PRAKTILINE OSA
Tehnoloogiad ja tööriistad
Käesolev projekt on realiseeritud programmeerimiskeeles Python (versioon 3.10+), mis on masinõppe valdkonnas kõige laialdasemalt kasutatav keel tänu rikkalikule ökosüsteemile ja kogukonna toetusele.
Andmetöötluse teek – Polars. Tavapärase Pandas'i asemel kasutatakse suure jõudlusega DataFrame'i teeki Polars, mis on kirjutatud programmeerimiskeeles Rust ning toetab andmete laiska hindamist (lazy evaluation) ja automaatset paralleliseerimist kõigil protsessori tuumadel. See valik võimaldab töödelda kümneid miljoneid ridu märkimisväärselt kiiremini kui Pandas.
Masinõppe raamistik – LightGBM. Mudelite treenimiseks kasutatakse Microsoft'i gradient-boosting'u teeki LightGBM (Light Gradient Boosting Machine), mis on spetsialiseeritud histogrammipõhisele otsustuspuu ansambli treenimisele. LightGBM on tuntud oma kiiruse, väikese mälukasutuse ja kõrge täpsuse poolest tabelkujuliste andmete peal — omadused, mis teevad selle sobivaks finantsandmete analüüsiks.
Hüperparameetrite optimeerimine – Optuna. Mudelite hüperparameetrite leidmiseks kasutatakse Bayesi optimeerimise raamistikku Optuna, mis kasutab TPE (Tree-structured Parzen Estimator) valimit koos katsete varajase lõpetamisega (pruning) Hyperband-algoritmiga. See võimaldab leida optimaalsed parameetrid kiiremini kui jõumeetod (grid search).
Tuletisinstrumentide kiirjäreldamine – ONNX. Treenitud mudelid konverteeritakse ONNX (Open Neural Network Exchange) formaati, mis võimaldab tarkvarasõltumatut kiirjäreldamist tootmiskeskkonnas ONNX Runtime'i abil.
Paralleliseerimise tööriistad. JIT-kompileerimisteek Numba (@njit, prange) kasutatakse arvutuslikult mahukate andmepuhastusoperatsioonide teostamiseks kõigil protsessori tuumadel samaaegselt. Joblib pakub kõrgema taseme paralleliseerimist mudelite treenimiseks ja andmete eeltöötluseks.
Riskiarvutuste teek – SciPy. Statistilised riskimõõdikud (protsentiilarvestused, jaotuse parameetrid) arvutatakse SciPy teegi abil.
Andmeallikas – Binance Futures API. Ajaloolised turuandmed on kogutud Binance'i alati-saadaoleva futures-turu REST-liidese kaudu. Andmekoguja toetab nii privaatset (API võtmega) kui ka avalikku (võtmeta) töörežiimi.
Seadistuste haldus. Kõik süsteemi parameetrid (protsessorite kasutusprotsent, RAM-piirangud, LightGBM'i hüperparameetrite vaikeväärtused, riskipiirid) on tsentraalselt hallatud YAML-seadistusfailis (config.yaml), mis muudab süsteemi kohandatavaks ilma lähtekoodi muutmata.
Projekti struktuur
Projekt jaguneb kaheks tippkaustaks:
aiTrainCrypto/
├── data/              ← Ajaloolised turuandmed CSV-formaadis
└── project/           ← Kogu süsteemi lähtekood
data — Sisaldab Binance Futures'i turuandmeid CSV-failidena. Iga fail vastab ühele varasümbolile ja andmetüübile: hinnaküünlad (klines) viiel ajavahemikul (5m, 15m, 1h, 4h, 1d), finantseerimismäärad (funding rate) ja avatud intressid (open interest). Andmed hõlmavad kuni 100 erinevat USDT-marginaaliga perpetual futures lepingut.
data — Andmekogumine ja laadimine. Sisaldab BinanceDataCollector klassi, mis kogub andmeid Binance API kaudu, ning kõrgejõudluslikke Polars-põhiseid laadureid (PolarsDataLoader) paraleelseks CSV-lugemiseks.
features — Tunnusearvutus (feature engineering) mudeli tüübi järgi:
scalp_features.py — 5m/15m ajavahemiku lühiajalised tunnused
intraday_features.py — 1h ajavahemiku päevasisesed tunnused
swing_features.py — 1d/4h pikaajalised tunnused
fast_features.py — kõigil mudelitel kasutatavad baastunnused
training — Mudelite treenimise skriptid mudeli tüübi järgi (train_scalp.py, train_intraday.py, train_swing.py, train_risk.py). Iga skript laadib andmed, arvutab tunnused, jagab andmestiku aegrea lõike (TimeSeriesSplit) järgi ning treenib LightGBM mudeli.
risk — Riskihalduse moodul:
cvar_engine.py — CVaR (Conditional Value at Risk) kalkulaator
kelly_sizing.py — Kelly kriteeriumipõhine positsioonide suurustamine
drawdown_controller.py — Dünaamiline riskikontroll maha-tõmbe järgi
risk_model.py — Tsentraalne riskimootor, millel on vetovõim kõigi tehingute üle
train_risk_optuna.py — Riskimudeli treenimine Optunaga
meta — Meta-otsustusмootor (meta_engine.py), mis ühendab kolme mudeli signaalid kaalutud hääletusreeglitega suuna ja usalduse lõpliku otsuse saamiseks. Sisaldab ka mündi edetabeli mooduli (ranking.py), mis järjestab varasid tugevuse ja korrelatsiooni järgi.
signals — Signaaligeneraator (signal_generator.py), mis teisendab meta-otsuse konkreetseks kauplemissignaaliks koos sisenemishinnaga, stoppkahjumi, kasumivõtu tasemete, finantsvõimenduse ja positsiooni suurusega.
validation — 12-kontrollpunktiga valideerimistraar (run_validation.py), mis katab andmete terviklikkust, tunnuste stabiilsust, ülesobitamise tuvastamist, ONNX ekspordi kontrollimist ja tootmisvalmiduse hindamist.
models — Salvestatud treenitud mudelid (.pkl ja .onnx formaadis).
logs — Käivituslogid ja auditijäljed riskimootorist JSONL-formaadis.
config.yaml — Tsentraalne seadistusfail kõigi süsteemiparameetrite jaoks.
train_with_risk.py — Monoliitne treenimis- ja riskimoodul, mis sisaldab kõiki komponente ühes failis kiirekäivitamiseks. Sisaldab PolarsDataLoader, PolarsFeatureEngine, CVaRCalculator, KellyCalculator, DrawdownMonitor, RiskEngine ja LightGBMTradingModel klasse.
 
Andmete töötlemine
 Andmete kogumine
Andmed kogutakse Binance Futures'i REST-liidese kaudu klassi “BinanceDataCollector“ abil. Iga varasümboli kohta kogutakse järgmised andmetüübid:
	OHLCV hinnaküünlad viiel ajavahemikul: 5 minutit, 15 minutit, 1 tund, 4 tundi ja 1 ööpäev. Iga rida sisaldab avamishinda, maksimumhinda, minimumhinda, sulgemishinda ja käivet (Open, High, Low, Close, Volume).
	Finantseerimismäärad (funding rate) — positiivseid ja negatiivseid igakvaddrandilisi tasusid, mis kajastavad pikka-lühikeste positsioonide tasakaalu turul.
	Avatud intressid (open interest) 5-minutiliste intervallidena, mis näitavad avatud lepingute kogusummat.
Andmekoguja rakendab automaatset vigade käsitlemist ja uuesti-proovimist API vastuste tükelüksuse piirangute (rate limiting) korral.
 Andmete laadimine
Suure andmemahu tõttu kasutatakse Polars'e laisklaadimist (pl.scan_csv), mis ei loe andmeid mällu, vaid konstrueerib täitmisplaani. Andmed loetakse füüsiliselt mällu alles arvutusahela lõpus .collect() kutsega. See vähendab mälukasutust märkimisväärselt, kuna iga vahetulemuse salvestamist välditakse.
Skeemi ühtsuse tagamiseks kehtestatakse laadimisel veergude andmetüübid ette (schema_overrides): kõik hinna- ja mahu-veerud teisendatakse Float64-tüüpi, mis välistab tüübikonflikte erinevate sümbolite failidevahel.
 Tunnusearvutus
PolarsFeatureEngine klass arvutab igale sümbolile ja ajavahemikule üle 150 tehnilise tunnuse ühes Polars-väljenditahvas (expression chain), mis käivitatakse paralleelselt kõigil protsessori tuumadel:
	Hinna tunnused: liikuvad keskmised (SMA, EMA) perioodidel 3–200, logaritmilised tulud (log returns), volatiilsuse mõõdikud.
	RSI (Relative Strength Index) kuuel perioodil (5, 7, 9, 14, 21, 28).
	MACD (Moving Average Convergence Divergence) kolmes konfiguratsioonis koos signaali- ja histogrammijoonega.
	Bollingeri ribad kolmel perioodil (10, 20, 50) ja nelja standardhälbe kordajaga.
	ATR (Average True Range) ning normaliseeritud ATR volatiilsuse hindamiseks.
	Mahumõõdikud: mahu liikuvad keskmised, mahu anomaaliate tuvastus, OBV (On-Balance Volume) ja selle nõlvakus.
	Statistilised tunnused: hinde z-skoor (z-score), libisev kallutatavus (rolling skewness), realiseeritud volatiilsus.
	Küünlavarju mustrid: keha suurus, ülemine ja alumine vari, pullish/bearish indikaator, doji tuvastus.
	Hilinemisfunktsioonid (lag features): sulgemishind ja tulu 1, 2, 3, 5 ja 10 perioodi tagant.
	Riskitunnused: rolling VaR (Value at Risk), maha-tõmbe (drawdown) joondiagramm, tagasipöördumisvolatiilsus.
Kõik NaN- ja lõpmatu (Inf) väärtused asendatakse nulliga Polars-väljenditega enne NumPy massiivi teisendamist, vältides andmete saastumist.
 Sildimine
Iga rea sihtmuutuja (target) on kolme klassiga kategooriline tunnus:
	0 — hind langeb järgmisel perioodil üle 0,2% (Down)
	1 — hind jääb ±0,2% vahemikku (Flat)
	2 — hind tõuseb järgmisel perioodil üle 0,2% (Up)
See kolmeklassiline lähenemisviis on sobivam kauplemise kontekstis kui binaarne klassifikatsioon, kuna see võimaldab mudelil eristada selget suundumust tasasest turust ning vähendada vale positiivseid signaale.
 Andmestiku jaotamine
Kuna tegu on aegridade andmetega, on andmestiku jaotamisel hädavajalik säilitada ajalist järjestust ning vältida tulevikupõhist andmeleket (look-ahead bias). Kasutatakse TimeSeriesSplit meetodit: treenimine ja valideerimine toimuvad rangetel ajalõikudel, kus valideerimisandmed on alati hilisemad kui treenimisandmed. Standardne jaotamine on 70% treenimine, 10% valideerimine, 20% testimine.
Masinõppe mudelite treenimine
 Mudelite arhitektuur
Süsteem implementeerib kolm eraldi prognoosimudelit, millest igaüks on spetsialiseeritud erinevale kauplemishorisondile:
Mudel	Ajavahemik	Eesmärk
Scalp	5 minutit	Lühiajaline (minutid–tunnid)
Intraday	1 tund	Päevasisene (tunnid–päev)
Swing	1 ööpäev	Keskmise pikkusega (päevad–nädalad)

Lisaks treenitakse eraldi riskimudel, mis hindab olukorra riskitaset ja mida kasutatakse riskimootori poolt tehingute kontrollimiseks, mitte otse signaalide genereerimiseks.
Kõik mudelid põhinevad LightGBMTradingModel klassil, mis laiendab abstraktset baasklassi BaseTradingModel. LightGBM sobib finantsandmete jaoks eelkõige seetõttu, et see:
	töötleb tabelkujulisi andmeid kiiresti histogrammipõhise puukonstruktsiooni abil
	on robustne müra ja puuduvate väärtuste suhtes
	pakub sisseehitatud tunnusetähtsuste analüüsi (feature importance)
	toetab varast peatamist (early stopping) ülesobitamise välistamiseks
 Treenimisprotsess
Iga mudeli treenimine koosneb järgmistest sammudest:
	Andmete laadimine ja tunnuste arvutamine — vastava ajavahemiku andmed laaditakse PolarsDataLoader'i abil ning PolarsFeatureEngine arvutab kõik tunnused paralleelselt.
	Andmete puhastamine ja normaliseerimine — NaN- ja Inf-väärtused eemaldatakse Numba JIT-kompileeritud funktsioonidega (parallel_nan_to_num), mis töötab kõigil protsessori tuumadel samaaegselt. Normaliseerimine toimub Polars-põhise parallel_robust_scale funktsiooniga, mis arvutab mediaani ja interkvartiilsete vahemike (IQR) alusel. Erinevalt StandardScaler'ist on RobustScaler statistiliselt erandlikele (outlier) väärtustele vastupidavam, mis on finantsandmetes oluline.
	Ristvalideerimine — LightGBM mudelit treenitakse TimeSeriesSplit lõiketel varajase peatamisega. Iga lõike AUC (Area Under the Curve) on salvestatud, et tuvastada mudeli stabiilsus ajas.
	Mitmeobjektiivne kahjufunktsioon — lisaks standardsele klassifikatsioonikahjumile (log-loss) rakendatakse CVaR-põhist kaotusmõõdikut, mis täiendavalt karistab mudelit varade sabariski eest. See integreeritud lähenemine loob seose prognoosiheaduse ja rahalise riski vahel juba treenimisfaasis.
	Hüperparameetrite optimeerimine — Optuna treenitab iga mudeli jaoks eraldi uurimust (study), katsetades 20–50 parameetrite kombinatsiooni (vastavalt config.yaml seadistusele) ja valides parima konfiguratsiooni.
	Mudeli salvestamine — lõplik mudel salvestatakse nii .pkl (Joblib) kui ka .onnx formaadis. ONNX formaat on mõeldud tootmiskeskkonna kiirjäreldamiseks, kus puudub vajadus täieliku Pythoni keskkonna järele.
 Tunnusetähtsus
Pärast treenimist analüüsitakse LightGBM'i tunnusetähtsuste pingerida, mis näitab, millised tunnused mõjutasid mudeli otsuseid enim. See analüüs võimaldab eemaldada mürast tunnused ja vähendada mudeli keerukust ilma täpsust kaotamata.
Riskihalduse süsteem
Riskimootor on süsteemi TSENTRAALNE KONTROLLER, millel on vetovõim kõigi kauplemissignaalide üle. Ükski signaal ei jõua rakendamisse ilma riskimootori heakskiiduta. Mootor integreerib kolm sõltumatut riskimõõdikut, mille kombinatsioon tagab mitmetasandilise kaitse.
 CVaR mootor (Conditional Value at Risk)
CVaR, tuntud ka kui Expected Shortfall (ES), mõõdab oodatavat kahjumit halvimatel juhtudel. Erinevalt VaR (Value at Risk) mõõdikust, mis näitab ainult piirtaset, arvutab CVaR kahju keskmist väärtust kõige halvema \alpha osakaalu jaoks.
CVaR_\alpha=-\frac{1}{1-a}\int_{a}^{1}{q_{U\ }du}
kus q_u on taseme u kvantiilfunktsioon kahjujaotusest.
Süsteem kasutab 95% usaldustaset ja 252 perioodi ajaloolist akent. CVaR-kalkulaator täidab kolm rolli:
	Kahjufunktsiooni komponent — CVaR-põhine trahvitermin lisatakse treenimisfaasi kahjufunktsioonile, suunates mudeli vältima sabasündmusi (tail events).
	Tehingute lubamisfilter — enne iga tehingut hinnatakse tehingu eeldatavat CVaR-mõju. Kui projitseeritud portfelli CVaR ületab läve (vaikimisi 3%), lükatakse tehing tagasi.
	Stressistsenaarium — stress-CVaR arvutatakse negatiivsete tagastuste kahekordistamisega, testides süsteemi käitumist kriisiolukorras.
 Kelly kriteerium (positsioonide suurustamine)
Kelly kriteerium määrab matemaatiliselt optimaalse panuse suuruse, et maksimeerida portfelli pikaajalise geomeetrilise kasvu.
f^\ast=\frac{p\cdot b-q}{b}
kus f^\ast on optimaalne portfelliosa, p on võidu tõenäosus, q=1-p on kaotuse tõenäosus ja b on väljamaksekordaja (payoff ratio).
Tooreid Kelly väärtusi kasutatakse finantsturul harva, kuna need on volatiilsed ja ohtlikud. Süsteem rakendab vraktsionaalset Kelly'd koos mitme kitsendusega:
	Dampenimine: kelly_dampened = kelly_raw × 0.25 (veerand-Kelly)
	Usalduse skaleerimine: kelly_scaled = kelly_dampened × confidence²
	Volatiilsuse kohandamine: madal volatiilsus → ×1.2, äärmuslik volatiilsus → ×0.2
	Riskiseisundi kohandamine: REDUCED seisund → ×0.5, OFF seisund → ×0
	CVaR ärakasutuse trahv: kui CVaR ärakasus > 50%, vähendatakse positsioonisuurust lineaarselt
	Kõvad piirid: lõplik väärtus piirati [0.01, 0.50] vahemikku
 Maha-tõmbe kontroller (Drawdown Controller)
Maha-tõmbe kontroller jälgib portfelli väärtuse langust haripunktist ja reageerib graduaalselt:
Maha-tõmbe tase	Riskiseisund	Positsioonide kordaja
< 5%	Täisrisk	1.00
5–10%	Hoiatus	0.75–1.00 (lineaarne)
10–15%	Kriitiline	0.25
≥ 15%	Hädaseisund	0.00 (kauplemine peatatud)

Lisaks rakendatakse mudelipõhiseid läve:
	Scalpimise mudel deaktiveeritakse maha-tõmbe > 8% korral
	Intraday mudel deaktiveeritakse maha-tõmbe > 12% korral
	Swing mudel deaktiveeritakse maha-tõmbe > 18% korral
Mudel taasaktiiveeritakse, kui portfell on 80% võrra haripunktile lähenenud.
Erakorralise kaitse tagamiseks sisaldab süsteem kill-switch mehhanismi, mis deaktiveerib viivitamatult kogu kauplemise ja mida saab käivitada programmaatiliselt.
 Riskimooduli otsustusahel
Iga kauplemissignaali hindame järgmises järjestuses:
Kill-switch aktiivne?	→  	VETOED
Mudel deaktiveeritud?	→	VETOED
Riskiseisund OFF?	→	REJECTED
CVaR piir ületatud?	→	REJECTED
Kelly	→	positsionisuurus
Drawdown kordaja	→	lõplik suurus
Suurus < 0.1%?	→	REJECTED
Suurus < 50% Kelly'st	→	REDUCED
Muul juhul	→	APPROVED

Kõik otsused logitakse JSON Lines formaadis auditijäljeks, mis sisaldab otsuse põhjendust, CVaR väärtust, Kelly osakaalu ja hetke maha-tõmmet.
Valideerimine ja optimeerimine
 12-kontrollpunktiga valideerimiskonveier
run_validation.py orkestreerub 12 sõltumatut kontrollpunkti, mis katavad kogu süsteemi elutsükli:
Nr	Kontrollpunkt	Eesmärk
1	Andmete terviklikkus	NaN/outlier-ide kontroll, ajaline järjestus, andmelekke tuvastamine
2	Tunnuste valideerimine	Tunnuste jaotus mudelite lõikes, multikollineaarsus, stabiilsus
3	Siltide valideerimine	Sihtmuutuja definitsioonide kontroll, horisondi valideerimine
4	Baasmeetmed	Juhusliku ja konstantse klassifikaatori tulemuste salvestamine võrdluseks
5	Optuna optimeerimine	Mudelipõhine hüperparameetrite otsing
6	Riskimudeli valideerimine	CVaR, Kelly ja maha-tõmbe kontrollerite kontroll
7	Ansambli valideerimine	Kolme mudeli koosmõju ja konsensuse hindamine
8	Ülesobitamise kontroll	Train/val/test meetrikate erinevuse analüüs, OOS (out-of-sample) halvenemine
9	ONNX valideerimine	Kontrollitakse, et ONNX mudeli ennustused vastavad LightGBM tulemustele
10	Tootmisvalmidus	Failide, konfiguratsioonide ja sõltuvuste kontroll
11	Logimine ja jälgitavus	Auditijälje täielikkuse kontroll
12	Lõplik ülevaade	Kogu süsteemi kokkuvõttev hinnang

Ülesobitamise tuvastamine
OverfittingController hindab mudeli üldistamisvõimet mitme kriteeriumi alusel:
	Train/val/test AUC erinevus peab jääma alla 5%
	Vigade jaotus treenimis- ja valideerimiskogumil peab olema statistiliselt sarnane (Kolmogorov–Smirnovi test)
	Mudeli täpsus mitme ajalõike lõikes peab olema ühtlane
	Out-of-sample halvenemine, võrreldes valideerimisandmetega, ei tohi ületada 10%
 Optuna hüperparameetrite optimeerimine
Optuna loob iga mudeli tüübi jaoks eraldi uurimuse (study). Iga katse (trial) hindab järgmisi parameetreid:
	num_leaves (31–511), max_depth (4–15), min_data_in_leaf (10–200)
	learning_rate (0.005–0.3, logaritmiline skaala), num_boost_round (500–5000)
	feature_fraction (0.5–1.0), bagging_fraction (0.5–1.0)
	reg_alpha (L1 regularisatsioon), reg_lambda (L2 regularisatsioon)
Optimeerimisobjektiivseks on mitmeobjektiivselt kaalutud skoor, mis seob AUC, CVaR, maksimaalse maha-tõmbe, Sharpe'i ja Sortino suhte üheks skalaariks:
skoor\ =0.15\cdot AUC-0.20\cdot\mid CVaR\mid-0.20\cdot MaxDD+0.15\cdot Sharpe+0.10\cdot Sortino+0.10\cdot stabiilsus\bigm\bigmMediaanprunimine (MedianPruner) lõpetab halvad katsed vara, säästa arvutusaega
 Baasmudelite võrdlus
Mudelite hindamiseks realistlikus kontekstis treenitakse neli baasmeedet (baseline):
	Juhuslik klassifikation — ennustused valitakse juhuslikult → AUC ≈ 0.50
	Konstantne ennustus — alati ennustatakse enim esinevat klassi
	Loogiline regressioon — lineaarne klassifikaator
	Otsustuspuu — üks madala sügavusega puu
LightGBM mudelite tulemus peab ükskõik millise baasmudelite tulemuse selgelt ületama, kinnitades, et mudel on õppinud reaalset mustrit, mitte kohanud statistilisi artefakte.
 Signaalide genereerimine
Valminud süsteemi täitmisvoo lõpus genereerib SignalGenerator inimloetavad kauplemissignaalid. Protsess on järgmine:
	Mudelite ennustused — kõik kolm mudelit arvutavad tõenäosuste vektori [P_down, P_flat, P_up].
	Meta-mootor — MetaDecisionEngine ühendab signaalid kaalutud valemiga: score_long = 0.5·P_up_scalp + 0.3·P_up_intraday + 0.2·P_up_swing. Tehing avaldatakse ainult juhul, kui skoor ületab läve (vaikimisi 0.65) ja swing-mudel ei ole vastassuundne.
	Riskimootor — RiskEngine hindab signaali CVaR kontekstis, arvutab Kelly-põhise positsioonisuuruse ja rakendab maha-tõmbe kordaja.
	TradingSignal objekt — heakskiidetud tehingust genereeritakse struktuur, mis sisaldab: sümbolit, suunda (LONG/SHORT), usaldust, sisenemishinda ja -vahemikku, stoppkahjumi taset, kahte kasumivõtu taset, finantsvõimendust, positsioonisuurust protsendina portfellist ning signaali kehtivusaega.
Signaali formaat JSON kujul:
 
Kogu süsteemi arhitektuuri kokkuvõtvalt iseloomustab kolmekordne kaitseliin: tunnusepõhine masinõppe ennustus, mitmeobjektiivne optimeerimine finantsriski arvestusega treenimisfaasis ning reaalajaline riskimootor, mis on võimeline iga üksiku tehingu kontekstuaalseks hindamiseks ja tagasilükkamiseks. Selline modulaarne ülesehitus tagab süsteemi läbipaistvuse, testitavuse ja kohandatavuse erinevate turutingimuste jaoks.

