# BIST Akıllı Yatırım Asistanı

Borsa İstanbul (BIST) ve ABD piyasaları için hisse analizi ve portföy yönetimi platformu. Hisseleri teknik analiz, haber duyarlılığı, analist primi ve değerlemeyi birleştirerek 0–100 arası puanlar; ardından portföy önerir, onları günlük yönetir ve **bunların gerçekten işe yarayıp yaramadığını ölçer.**

Canlı demo: [bist-analyzer-wrtwzcsmdv4z9fxrpkkrcz.streamlit.app](https://bist-analyzer-wrtwzcsmdv4z9fxrpkkrcz.streamlit.app)

> **Yatırım tavsiyesi değildir.** Bu bir araştırma ve eğitim projesidir. Bkz. [Sorumluluk reddi](#sorumluluk-reddi).

---

## Bu projeyi farklı kılan ne

Çoğu hisse tarayıcısı ne alınacağını söyler. Asıl zor soru, *bunu söyleyen şeyin işe yarayıp yaramadığıdır* — emeğin çoğu buraya gitti:

- **Ağırlıklar tahmin değil, ölçüm.** Bileşen ağırlıkları, tarihe sadık (point-in-time) panellerde ölçülen kesitsel Bilgi Katsayılarından (Grinold–Kahn) gelir.
- **Bilimsel kontrol grupları.** Her sonuç; aynı evrenden rastgele çekilen 6 hisselik portföylerle, eşit ağırlıklı pasif sepetle, endeksle ve ders kitabı 12-1 momentum portföyüyle karşılaştırılır.
- **Yönetim, seçimden ayrı ölçülür.** Her haftalık canlı kohort iki kez kaydedilir: biri pasif, biri aynı seçim ve fiyatlarla aktif yönetilen. Aradaki fark yönetimin katkısıdır.
- **Kilitli veri kasası.** Son iki yıl varsayılan olarak her backtest'in dışında tutulur ve yalnız bir kez açılabilir; o koşunun hipotezleri önceden depoya yazılır.
- **Çoklu test düzeltmesi.** Deflated Sharpe Ratio, gerçekten denenen yapılandırma sayısına (~695) göre hesaplanır.
- **Olumsuz sonuçlar da yazılır.** Burada test edilen hipotezlerin çoğu başarısız oldu ve raporlar bunu açıkça söyler — projenin kendi manşet iddialarını zayıflatanlar dahil.

---

## Mimari

| Dosya | Satır | Görev |
|---|---|---|
| `bist_analyzer.py` | ~12.700 | Kullanıcıya dönük her şey: veri çekme, skor motorları, portföy yöneticisi, günlük yönetim, denetim, depolama ve tüm Streamlit sayfaları |
| `news_engine.py` | ~2.300 | Bağımsız haber duyarlılık motoru (ana dosyaya geri bağımlılığı yok) |
| `pipeline_backtest.py` | ~2.350 | Tam hat tarihsel simülasyon, kontroller, walk-forward, transfer testleri, DSR, veri kasası |
| `weight_calibration.py` | ~700 | Point-in-time paneller kurar, ağırlıklar için IC hesaplar |
| `bilanco.py` | ~600 | Çeyreklik mali tablolar (TMS 29 farkında, sabit TÜFE parası) ve ders notu oranları |
| `skor_gunlugu.py` | ~330 | Tüm BIST evreni için her skor bileşeninin günlük kalıcı kaydı |
| `daily_robot.py` | ~310 | Otonom günlük bakım (GitHub Actions'ta çalışır) |
| `tools/` | — | KAP bildirim geçmişi, olay çalışması, ifade ölçümü, sınırlı veritabanı bakımı |

### Çekirdek motorlar

`PortfolioScanner` dinamik evreni tarar → `PortfolioManager` vade × yatırımcı profiline göre portföy önerir → `PortfolioDoctor` onları günlük yönetir → `PortfolioAudit` kuralların işe yarayıp yaramadığını ölçer.

Evren dinamik ve likidite filtrelidir, her gün yenilenir: TL hacim eşiğini aşan ~500 BIST hissesi ve günlük $50M hacmi aşan ~1.400 ABD hissesi.

---

## Skorlama modeli

| Bileşen | Ağırlık | Kaynak |
|---|---|---|
| Teknik | %60 | SMA, RSI, MACD, Bollinger, Stochastic, OBV, ADX, 52 hafta pozisyonu |
| Haber duyarlılığı | %10 | Kademeli ağırlıklı RSS kaynakları + KAP bildirimleri; skoru yalnız fiyatla doğrulanmış yönler oynatabilir |
| Prim potansiyeli | %20 | Analist konsensüs hedefi / güncel fiyat (TradingView, en az 3 analist) |
| Değerleme | %10 | Sektöre göre F/K, PD/DD ve özsermaye kârlılığı |

Bir bileşenin gerçek girdisi yoksa (haber yok, analist hedefi yok) ağırlığı nötr 50 ile ortalanmaz, tekniğe aktarılır. Duyarlılık %25'ten %10'a indirildi: denetim, hisselerin çoğunun tam 50 aldığını ve bunun sıralamaya bilgi katmadan skorun yayılımını daralttığını gösterdi.

**Sinyal bantları:** 72+ Güçlü Al · 57–71 Al · 43–56 Nötr · 30–42 Sat · <30 Güçlü Sat

### Teknik stil rejime bağlı

İlk kontraryan skorlama (aşırı satılmış RSI'ı, düşük Bollinger konumunu, 52 hafta zirvesinden uzaklığı ödüllendiren) BIST'te **sıfır öngörü gücü** ölçtü. Piyasa momentum rejiminde: yüksek RSI ve yüksek 52 hafta pozisyonu *daha yüksek* ileri getiri öngördü. Canlı ve backtest skor yolları artık momentum stilini kullanır.

### Skoru oynatan haber, fiyatı da oynatmış olmalı

2018–2024 arası 411.656 KAP bildirimi bir olay çalışmasıyla fiyatlarla eşleştirildi. 16 yön hipotezinden ikisi ayakta kaldı: **pay geri alımı** (30 günde piyasaya göre %+2,63, t = 4,6) ve **bedelsiz sermaye artırımı** (%−3,86, t = −5,0 — sulandırma teorisinin beklediğinin tersi). Etkisi gürültü çıkan ifadeler artık oy kullanmıyor; "olumlu" etiketi ancak yönü kanıtlıysa skoru yükseltiyor.

---

## Portföy Yöneticisi

Tüm evren taramasından **vade** (kısa / orta / uzun) × **yatırımcı profili** (Temkinli / Dengeli / Agresif) bazında portföy önerir: sektör sınırı, korelasyon filtresi, ATR'ye ters orantılı ağırlık ve tek pozisyon üst sınırıyla.

- **Piyasa rejimi kapısı.** BIST: XU100 trendi + BIST-30 genişliği + kur baskısı. ABD: S&P 500 trendi + genişlik + *seviye olarak* VIX.
- **Asgari çeşitlendirme koruması.** Filtreleri üçten az hisse geçerse portföy kurulmaz.
- **Performans defteri.** Her portföy için günlük NAV; nominal + ENAG-reel + endekse göreli, Sharpe, Sortino, en büyük düşüş ve Bilgi Oranı ile.

### Günlük aktif yönetim

`PortfolioDoctor` her iş günü kapanıştan sonra çalışır:

1. **Çıkışlar günlük ve bütçesiz** — stop kırılımı, iz süren stop, tez bozulması (skor çöküşü). Risk azaltmak hiçbir zaman sınırlanmaz.
2. **Pozisyon başına tek ağırlık kararı** — hedefte bir kerelik kısmi kâr alımı, ardından yoğunlaşma sınırı.
3. **Alım aylık** — her ayın ilk koşusunda, o ayın taze seçimlerine doğru; düşüş freni ve ayı rejimi kapısıyla. Bu, backtest'in doğruladığı yapılandırmadır; önceki günlük alım sürümü yılda yaklaşık 14 puan daha kötü ölçüldü.

Nakit gerçek bir pozisyondur, işlem maliyeti NAV'dan düşülür ve her işlem gerekçesiyle denetim kaydına yazılır. `PortfolioAudit` stop isabetini, takas isabetini, maliyet yükünü ve yönetilen-eksi-pasif farkını ölçer; parametre değişikliğini yalnız *önerir*, uygulamak editör anahtarı ister.

---

## Kanıt ve doğrulama

Aşağıdaki tüm sayılar **geliştirme penceresine (2018-08-31 → 2024-08-31)** aittir; sonraki iki yıl veri kasasında kilitli. Bilerek göze hoş gelmeyen sayılar bunlar: kilitli dönemi de içeren eski beş yıllık pencere çok daha iyi görünüyordu, kasanın kilitlenme sebebi de tam olarak buydu.

### Seçim şanstan iyi mi?

Sistemin son NAV'ının, aynı evrenden aynı rebalans ve maliyetle çekilen 50 rastgele 6 hisselik portföy arasındaki yüzdeliği; parametreler BIST'te seçilip diğer piyasalara değiştirilmeden uygulandı:

| Piyasa | Yalnız seçim | Sonuç |
|---|---|---|
| BIST | 10. | rastgeleyi geçemiyor |
| ABD | 94. | rastgeleyi ve S&P 500'ü geçiyor |
| İngiltere | 22. | rastgeleyi geçemiyor (sıfır maliyet dahil her maliyet düzeyinde) |

Hiçbir parametresi ayarlanmamış **ders kitabı 12-1 momentum portföyüne** karşı sistemin alfası üç piyasada da sıfır ya da eksi; ABD'de (son NAV 12,19'a 4,55) ve İngiltere'de (2,29'a 0,68) momentum tek başına daha iyi. Skor fonksiyonunun düz momentuma üstünlüğü kanıtlanmış değil.

**Deflated Sharpe Ratio: 0,11–0,31**, eşik 0,95. Backtest tek başına bir üstünlük kanıtlayamıyor.

### Ayakta kalan: BIST'te yönetim katmanı

| BIST | Yalnız seçim | Yönetilen |
|---|---|---|
| Rastgele-6'ya göre yüzdelik | 10. | 62. |
| Yıllık reel getiri (ENAG) | −%10,0 | **+%4,8** |
| En büyük düşüş | −%69,9 | **−%25,8** |

Karşılaştırma için: aynı dönemde XU100'ün yıllık reel getirisi −%13,2.

Ardışık altı alt dönemde (2018–2026, kasa kurulmadan önce ölçüldü) yönetilen iz **6'nın 6'sında** daha küçük düşüş yaşadı. 2018 kur krizinde yönetimsiz iz %69,9 düşerken yönetilen iz %23,8 düştü. Aynı kurallar ABD'ye **taşınmadı**, orada hem getiriyi hem düşüşü kötüleştirdi; yani bu genel bir yöntem özelliği değil, piyasaya özgü bir risk katmanı.

### Kalibrasyon bulguları

| Piyasa | Bulgu |
|---|---|
| BIST | Momentum IC **+0,045 (t = 2,8)**. En güçlü tekil sinyal: 52 hafta pozisyonu. |
| ABD büyük şirketler | İki stilde de istikrarlı üstünlük yok. |
| ABD orta ölçek | Önceki "istikrarlı üstünlük" bulgusu, likidite dilimi düzeltilince **tekrarlanmadı**. |
| ABD, rejime göre | Boğa rejiminde 52 hafta pozisyonu IC **+0,074 (t = 8,6)**; ayı rejiminde momentum skoru anlamlı biçimde **ters** (t = −4,7). Ayı rejiminde tüm ABD önerileri engellenir. |

### Diğer testler

- **Parametre araması durduruldu.** 5 genişleyen katlı walk-forward önerilen değişikliklerin neredeyse hepsini reddetti; eğitimin "en iyisi" test edilen aralığın tamamında geziniyorsa yüzey gürültüdür.
- **Mali sağlamlık skoru** (ders notu oranları, 20 kriter, sektöre göreli): önceden kaydedildi, bir kez test edildi, yönetilen izde **başarısız** oldu. Yönetimsiz iz için ayrı bir hipotez kasa açılışına önceden kaydedildi.
- **Kapasite:** karekök piyasa etkisi modeliyle üstünlük ~1M TL altında büyük ölçüde korunuyor, ~50M TL'de bitiyor.

---

## Bilinen Sorunlar

Ekim 2026'daki kod ve hesap incelemesinde bulundu. Düzeltilenler Yol Haritası'nın sonunda; bunlar hâlâ açık.

**Raporlanan sayıları etkileyenler**
- **Enflasyon verisi kısmen tahmini.** ENAG 2020'den önce yoktu, bu yüzden 2017–2019 için resmi TÜFE kullanılıyor. 2020–2025'te aylık rakamı yayımlanmamış aylar, yıl toplamı ENAG'ın yayımladığı yıllığı tutacak tek bir oranla dolduruluyor. Girilen son aydan sonraki aylar, girilene kadar son 12 ayın ortalamasını kullanıyor.
- **Fiyat endeksi, toplam getiri portföyüyle karşılaştırılıyor.** Hisse panelleri temettüyü içeriyor; XU100, S&P 500 ve FTSE 100 içermiyor. "Endekse göre" sayılar ABD'de yılda ~2 puan, İngiltere ve BIST'te daha fazla lehimize.
- **Backtest yalnız teknik çekirdeği test ediyor.** Haber ve analist hedefi tarihsel olarak kurulamıyor; canlı bileşik skor ilk 6 seçimin yaklaşık 3'ünü backtest'le paylaşıyor. Canlı likidite filtresi de backtest'inkinden 20 kat gevşek.
- **Hayatta kalma yanlılığı:** evren bugünün likit listesi; simülasyonla reel yıllık getiride ~2 puanla sınırlandı.
- **En büyük düşüş nominal NAV üzerinden ölçülüyor**; yüksek enflasyonda reel düşüşü eksik gösterir.
- **Tek hisse backtest'inde düşüş** yalnız kapanan işlemlerden hesaplanıyor; işlem içi düşüşler sayılmıyor.
- **RSI basit ortalamayla hesaplanıyor**, Wilder yumuşatmasıyla değil; değerler TradingView'dan farklı. Kalibrasyon bu tanımla ölçüldüğü için olduğu gibi bırakıldı.

**Davranış ve sağlamlık**
- **Noktalı ABD hisse sınıfları** (örn. `BRK.A`) Yahoo'nun `BRK-A` biçimine çevrilmiyor ve sessizce atlanıyor.
- **Yerel SQLite için göç (migration) sistemi yok.** Çok eski bir sürümün oluşturduğu veritabanında sütun adları farklı olabilir (eski Zaman Makinesi tablolarında görüldü); sıfırdan kurulumda sorun yok.
- **Zaman Makinesi** hâlâ eski portföy mantığını simüle ediyor, doğrulanmış hattı değil; bir koşu testte 10 dakikayı aştı ve iptal edilemiyor. Bu yüzden demo ziyaretçilerine kapalı.
- **`use_container_width`** ~85 yerde kullanılıyor; sabitlenmiş Streamlit 1.59'da çalışıyor ama kullanımdan kalkıyor, sürüm yükseltilince bozulur.
- **İngilizce arayüz çevirisi yarım.**
- **Canlı kanıt kısa.** Canlı kohortlar haftalardır çalışıyor ve dört yapılandırma kırılması taşıyor; yönetilen portföyler de aynı hisselere yakınsıyor (35 portföy 49 farklı hisse tutuyordu).
- **Alt bağımlılıklar serbest.** Doğrudan bağımlılıklar tam sabit; hash'li kilit dosyası henüz yok.

---

## Yol Haritası

**Sıradaki**
- Veri kasasını önceden kaydedilmiş testlerle bir kez açmak: kilitli iki yılda sistem–rastgele karşılaştırması ve yönetimsiz izde mali sağlamlık filtresi.
- Veri kaynağının izin verdiği yerde toplam getiri endeksleri.

**Birkaç aylık günlük bileşen kaydından sonra**
- Tarihsel olarak hiç doğrulanmamış duyarlılık, analist primi ve değerlemenin ileriye dönük IC'si. O zamana kadar ağırlıkları değişmez.

**Yöntem**
- Deflated Sharpe'ın tamamlayıcısı olarak kombinatoryal arındırılmış çapraz doğrulama ve backtest aşırı uyum olasılığı.
- Ders kitabı momentum portföyünü kalıcı kontrol olarak tutmak; skor fonksiyonunu yalnız rastgeleye değil ona karşı da test etmek.

**Ürün**
- Zaman Makinesi'ni tam hat backtest mantığıyla yeniden kurmak.
- ABD hisse sınıfı kodlarını eşlemek, `use_container_width`'ten çıkmak, İngilizce arayüzü bitirmek, hash'li kilit dosyası eklemek.

**Ekim 2026 incelemesinde düzeltilenler**
- Zarar eden ya da özkaynağı eksi şirket değerlemede *en yüksek* puanı alıyordu (eksi F/K ve PD/DD dev iskonto sayılıyordu).
- yfinance biçim değişikliğinden sonra Zaman Makinesi'nin endeks getirisi hep 0'dı; "alfa" aslında ham getiriydi.
- Portfolyum tek hisselik portföyü %−100 gösteriyordu (fiyat 0 okunuyordu); ABD listesi tek sembolü fiyatlayamıyordu.
- Bilgi Oranı enflasyondan arındırılmış portföy getirisinden nominal endeks getirisini çıkarıyordu; ABD (dolar) portföyleri Türk enflasyonuyla düşülüyordu.
- Eski yerel veritabanlarında analiz geçmişi kaydedilmiyordu; okuyan taraf yazan taraftan farklı bir sütun adı kullanıyordu.
- Portfolyum, sıfırdan kurulan veritabanında ilk hisse eklenince çöküyordu (aynı türden sütun adı uyuşmazlığı); hisse kartları da uzun zaman önce yanlışlıkla silinmişti, sayfada yalnız "Sil" düğmeleri görünüyordu.
- Sinyal Takip, dönemde SAT sinyali olup AL sinyali olmadığında çöküyordu.
- ENAG varsayılanlarında iki kez yazılmış bir ay, sıfırdan kurulumda resmi değeri bir tahminle eziyordu.
- **2020 öncesi enflasyon yanlıştı.** ENAG'ın 2020 öncesi verisi yok ve o aylar son yılın ortalamasıyla (~%20 yerine ~yıllık %54) dolduruluyordu. 2018–2024 penceresindeki her reel getiri yılda yaklaşık 5 puan düşük görünüyordu. Yönetilen BIST portföyü yılda −%0,8'den +%4,8'e çıktı; rastgele portföylere göre sıralama değişmedi. 2020–2024 yıllık toplamları da artık ENAG'ın yayımladığı rakamları tutuyor.

---

## Sayfalar

İki piyasa da aynı yedi sayfalık yapıdadır. ABD sayfaları, piyasa parametresi alan aynı render fonksiyonlarıdır; kopya değildir.

**Kokpit** (açılış — piyasa rejimi, makro şerit, günün sinyalleri, sistem durumu) · **Portföy Yöneticisi** · **Hisse Listesi** · **Hisse Analizi** · **Portfolyum** · **Backtest** · **Sinyal Takip**

İki tema (açık "Gazete", koyu "Obsidyen"). Piyasa `?piyasa=US`, tema `?tema=` ile bağlantıya eklenebilir.

---

## Otomasyon

`daily_robot.py` hafta içi iki kez GitHub Actions'ta çalışır: BIST kapanışından ve ABD kapanışından sonra. Her koşu NAV'ı günceller, haftalık gölge ve kontrol kohortlarını yeniler, günlük yönetimi ve kural denetimini çalıştırır, risk alarmlarını kontrol eder ve piyasa rejimini kaydeder. İkinci bir adım tüm BIST evreni için her skor bileşenini veritabanına yazar; bozuk günler sonradan ayıklanabilsin diye her koşunun kalite satırı da tutulur.

Robot işe başlamadan önce veritabanını yoklar ve hata varsa sıfırdan farklı kodla çıkar: daha önce bir hafta boyunca hiçbir şey yapmadan yeşil çalıştığı bir olay yaşandı, çünkü başarı hiç doğrulanmıyordu.

---

## Teknoloji

| Katman | Teknoloji |
|---|---|
| Arayüz | Streamlit + Plotly |
| Fiyat verisi | yfinance |
| Evren ve analist hedefleri | TradingView screener |
| Haber | feedparser (RSS kaynakları), bildirimler için borsapy ve KAP API |
| Mali tablolar | borsapy üzerinden İş Yatırım, resmi TÜFE ile sabit paraya çevrilmiş |
| Duyarlılık | Fiyatla doğrulanmış kural katmanları; isteğe bağlı Türkçe BERT |
| Depolama | Yapılandırılmışsa Turso (HTTP üzerinden libSQL), değilse yerel SQLite — ikisi arasında asla sessiz geçiş yok |
| Otomasyon | GitHub Actions |

---

## Kurulum

```bash
git clone https://github.com/Akif3445/bist-analyzer.git
cd bist-analyzer
pip install -r requirements.txt
streamlit run bist_analyzer.py
```

Bağımlılıklar tedarik zinciri önlemi olarak tam sürüme sabitlenmiştir: CI bunları ortamda veritabanı yazma anahtarı varken kurar, sabitlenmemiş ve ele geçirilmiş bir sürüm anahtarı sızdırabilir.

İsteğe bağlı `.env`:

```
ANTHROPIC_API_KEY=...      # yapay zekâ ile Türkçe özet
TURSO_DATABASE_URL=...     # bulut kalıcılığı; yoksa yerel SQLite
TURSO_AUTH_TOKEN=...
APP_EDIT_KEY=...           # yazma koruması (aşağıya bakın)
```

### Araştırma araçları

```bash
python pipeline_backtest.py run 8         # tam hat: XU100 / BIST-30 / rastgele-6 ile
python pipeline_backtest.py karsilastir 8 # yönetilen ve yönetimsiz
python pipeline_backtest.py walkforward 8 5
python pipeline_backtest.py momentum 8    # sistem ve ders kitabı momentumu, BIST/ABD/İngiltere
python pipeline_backtest.py dsr 8         # deflated Sharpe
python pipeline_backtest.py kasa          # veri kasası durumu
python tools/kap_olay.py --min-ornek 50   # KAP olay çalışması
python bilanco.py oran THYAO              # bir hissenin oran geçmişi
```

`CLAUDE.md` tam araştırma günlüğüdür: her ölçüm, her değişikliğin gerekçesi ve sonradan geri çekilen her sonuç.

---

## Güvenlik

Halka açık demo veritabanı yazma yetkisiyle çalışır ve Streamlit'in yerleşik kimlik doğrulaması yoktur. `APP_EDIT_KEY` tanımlıysa her arayüz yazma işlemi kenar çubuğundan kilit açmayı gerektirir; okuma herkese açık kalır. Ziyaretçiler tüm sistem portföylerini ve canlı performanslarını görür, ama **Portfolyum** her ziyaretçi için kişisel bir deneme alanına dönüşür: eklediği hisseler yalnız kendi tarayıcı oturumunda kalır, hiçbir yere kaydedilmez ve kimseye görünmez. Uzun süren Zaman Makinesi koşuları ziyaretçilere kapalıdır. HTML olarak gösterilen dış RSS içeriği kaçışlanır ve bağlantı şemaları kontrol edilir. SQL baştan sona parametrelidir. Günlük robot en az yetkiyle (`contents: read`) çalışır.

**Halka açık demoya gerçek kişisel varlık bilgisi girmeyin** — yazma koruması kirlenmeyi önler, ifşayı değil.

---

## Sorumluluk reddi

Yalnızca bilgilendirme ve eğitim amaçlıdır. Yatırım tavsiyesi değildir ve hiçbir menkul kıymet için al-sat önerisi içermez. Backtest istatistiksel olarak savunulabilir bir üstünlük kanıtlamıyor ve canlı kanıt haftalarla sınırlı. Kendi araştırmanızı yapın.

---

## Lisans

Tescilli. Tüm hakları saklıdır.
