# BIST Akıllı Yatırım Asistanı

Borsa İstanbul (BIST) ve ABD piyasaları için hisse analizi ve portföy yönetimi platformu. Hisseleri teknik analiz, haber duyarlılığı, analist primi ve değerlemeyi birleştirerek 0–100 arası puanlar; ardından portföy önerir, onları günlük yönetir ve **bunların gerçekten işe yarayıp yaramadığını ölçer.**

Canlı demo: [bist-analyzer.streamlit.app](https://bist-analyzer-wrtwzcsmdv4z9fxrpkkrcz.streamlit.app)

> **Yatırım tavsiyesi değildir.** Bu bir araştırma ve eğitim projesidir. Bkz. [Sorumluluk reddi](#sorumluluk-reddi).

---

## Bu projeyi farklı kılan ne

Çoğu hisse tarayıcı size ne alacağınızı söyler. Asıl zor soru, *size söyleyen şeyin çalışıp çalışmadığıdır* — emeğin büyük kısmı oraya gitti:

- **Ağırlıklar tahmin değil, kalibrasyon.** Bileşen ağırlıkları, geçmişin her noktasında hesaplanmış kesitsel Bilgi Katsayısı (Information Coefficient, Grinold–Kahn) ölçümlerinden geliyor.
- **Bilimsel kontrol grupları.** Sistem her hafta kendi seçimlerini gölge portföy olarak kaydeder — yanına iki kontrol grubu koyar: rastgele kura ile seçilmiş portföy ve eşit ağırlıklı pasif endeks sepeti. Sistem şansı ve pasifliği yenemiyorsa ortada bir üstünlük yoktur.
- **Yönetim, seçimden ayrı ölçülür.** Her haftalık kohort iki kez kaydedilir: biri pasif, diğeri aynı hisseler ve aynı fiyatlarla başlayıp aktif yönetilen. Aradaki fark **tam olarak yönetimin katkısıdır.**
- **Örneklem-dışı doğrulama.** Kural parametreleri bir dönemde seçilir, seçime hiç katılmayan başka bir dönemde sınanır. İç-örneklem optimizasyonunun ürettiği önerilerin yarısı bu testte gürültü çıkıp elendi.
- **Nominal değil, reel getiri.** Yüksek enflasyonlu bir ekonomide nominal getiri neredeyse anlamsızdır; performans ENAG enflasyonuna ve endekse göre raporlanır.
- **Bilinen yanlılıklar gizlenmez, yazılır.** Hayatta kalma yanlılığı, boğa piyasası örneklemi, iç-örneklem seçimi ve backtest ile canlı sistem arasındaki fark açıkça belgelenmiştir.

---

## Mimari

| Dosya | Satır | Görev |
|---|---|---|
| `bist_analyzer.py` | ~12.600 | Her şey: veri çekme, skorlama motorları, portföy yöneticisi, günlük yönetim, denetim, veritabanı, backtest ve tüm Streamlit sayfaları |
| `news_engine.py` | ~1.950 | Bağımsız haber duyarlılık motoru (ana dosyaya geri bağımlılığı yok) |
| `weight_calibration.py` | ~690 | Araştırma aracı: point-in-time paneller kurar, skorlama ağırlıklarını verilendirmek için IC hesaplar |
| `pipeline_backtest.py` | ~650 | Tam-hat tarihsel simülasyon, yönetim kuralı taraması, örneklem-dışı doğrulama |
| `daily_robot.py` | ~240 | Otonom günlük bakım (GitHub Actions üzerinde çalışır) |
| `tools/pmdb_admin.py` | ~90 | Sınırlandırılmış veritabanı bakım komutları |

### Çekirdek motorlar

`PortfolioScanner` dinamik evreni tarar → `PortfolioManager` vade × yatırımcı profiline göre portföy önerir → `PortfolioDoctor` onları günlük yönetir → `PortfolioAudit` kuralların işe yarayıp yaramadığını ölçer.

Evren dinamiktir ve likidite filtrelidir, günlük yenilenir: TL hacim eşiğini geçen ~500 BIST hissesi ve günlük 50M$ üzeri işlem gören ~1.400 ABD hissesi.

---

## Skorlama modeli

| Bileşen | Ağırlık | Kaynak |
|---|---|---|
| Teknik | %45 | SMA, RSI, MACD, Bollinger, Stochastic, OBV, ADX, 52 hafta pozisyonu |
| Haber duyarlılığı | %25 | Kademeli ağırlıklı RSS kaynakları + KAP bildirimleri |
| Yükseliş potansiyeli | %20 | Analist konsensüs hedefi ile güncel fiyat farkı |
| Değerleme | %10 | Sektör ortalamalarına göre F/K, PD/DD |

Ağırlıklar Temmuz 2026'da 35/35/20/10'dan güncellendi — kalibrasyon, haber duyarlılığının öngörü gücünün zayıf olduğunu gösterdi. Hedef fiyat veya temel veri yoksa o ağırlık boşta bırakılmaz, diğer bileşenlere dağıtılır.

**Sinyal bantları:** 72+ Güçlü Al · 57–71 Al · 43–56 Nötr · 30–42 Sat · <30 Güçlü Sat

### Teknik stil rejime bağlı

Orijinal kontrarian skorlama (aşırı satım RSI'ı, düşük Bollinger konumu, 52 hafta zirvesinden uzaklığı ödüllendiren) BIST'te **sıfır öngörü gücü** ölçtü. Piyasa momentum rejiminde: yüksek RSI ve yüksek 52 hafta pozisyonu *daha yüksek* ileri getiri öngördü (t > 3, alt dönemlerde tutarlı). Hem canlı hem backtest skorlama yolları artık momentum stilini kullanıyor.

Segment bazlı melez bir yaklaşım denendi ve **reddedildi** — farklı cetvelleri karıştırmak kesitsel sıralamayı bozuyor, IC +0.045'ten +0.012'ye düşüyor.

---

## Portföy Yöneticisi

Tüm evrenin taramasından, **vade** (kısa / orta / uzun) × **yatırımcı profili** (Temkinli / Dengeli / Agresif) matrisine göre portföy önerir. Sektör limiti, korelasyon eleme (>0.85 "aynı bahis" sayılır), ATR-ters-orantılı ağırlıklandırma ve tek pozisyon tavanı uygulanır.

- **Piyasa rejimi kapısı.** BIST: XU100 trendi + BIST-30 genişliği + kur baskısı. ABD: S&P 500 trendi + genişlik + **seviye olarak** VIX. Dolar bazlı portföyde kur riski yoktur, oynaklık rejimi vardır.
- **Minimum çeşitlendirme koruması.** Filtrelerden üçten az hisse geçerse portföy kurulmaz. Tek hisseye %100 ağırlık temkinlilik değil, konsantrasyondur.
- **Performans defteri.** Portföy başına günlük NAV; nominal + ENAG-reel + endekse göreli raporlanır, kurumsal metriklerle (Sharpe, Sortino, maksimum düşüş, Bilgi Oranı).

### Günlük aktif yönetim

`PortfolioDoctor` her hafta içi kapanıştan sonra çalışır ve gerçek bir portföy yöneticisi gibi davranır:

1. **Önce çıkışlar, bütçesiz** — stop kırılımı, takip eden stop, tez bozulması (skor çöküşü). Riski kesmek asla sınırlanmaz; sınırlanan sadece fırsat kovalamaktır.
2. **Pozisyon başına tek ağırlık kararı** — hedefte kısmi kâr alma, ardından yoğunlaşma tavanı.
3. **Nakit değerlendirme ve takas** — düşüş freni ve ayı rejimi kapısıyla sınırlı.

Nakit gerçek bir pozisyondur: uygun aday yoksa ağırlık %0 getiriyle nakitte bekler. Disiplin kuralları (minimum tutma süresi, satış sonrası bekleme, günlük işlem tavanı) savrulmayı önler ve **işlem maliyetleri NAV'a yansıtılır** — sürtünmesiz bir NAV aktif yönetimi olduğundan iyi gösterirdi.

Her aksiyon gerekçesiyle denetim defterine yazılır. `PortfolioAudit` ardından kuralların işleyip işlemediğini ölçer — stop isabeti, takas isabeti, maliyet yükü, nakit maliyeti ve yönetilen-eksi-pasif farkı — ve ancak yeterli örnek biriktiğinde parametre önerisi üretir. Öneriyi uygulamak editör anahtarı gerektirir; sistem kendi kendini yeniden ayarlamaz, çünkü on parametreli ve haftalarca veriye sahip sürekli bir öz-optimizasyon gürültüye uyum sağlar.

---

## Kanıt ve doğrulama

### Kalibrasyon bulguları

| Piyasa | Bulgu |
|---|---|
| BIST | Momentum IC **+0.045 (t = 2.8)**, alt dönemlerde tutarlı. En güçlü tekil sinyal: 52 hafta pozisyonu. |
| ABD mega-cap | Momentum IC +0.0065 (t = 0.53) — dünyanın en verimli segmentinde **üstünlük yok**. |
| ABD orta ölçek | Momentum IC **+0.0212 (t = 2.06)**, tutarlı. Üstünlük, analist takibinin seyreldiği yerde yaşıyor. |
| ABD, rejime göre | Boğa rejiminde ABD IC'si **+0.053'e (t = 6.0)** çıkıyor; ayı rejiminde momentum skoru anlamlı biçimde **ters dönüyor** (t = −3.24). Kısa vadeli öneriler buna göre engelleniyor. |

### Tam-hat backtest

Bugünün portföy kurma hattı, beş yıl boyunca point-in-time işletildi; aylık rebalans ve devir maliyeti dahil:

| | Nominal | **Reel (ENAG)** | Sharpe | Maks. düşüş |
|---|---|---|---|---|
| Aktif yönetilen | %+4083 | **%+10.9 CAGR** | 3.33 | −%25.8 |
| Yönetimsiz | %+3642 | %+8.4 CAGR | 2.14 | −%48.4 |
| XU100 endeksi | %+863 | **−%17.3 CAGR** | — | — |

Aynı dönemde enflasyon **25 kat**tı — nominal rakamların neden asla tek başına söylenmemesi gerektiğinin sebebi tam olarak budur.

### Örneklem-dışı doğrulama

Parametreler 8 yıllık pencerenin (2018 kur krizi ve 2020 çöküşünü içerir) ilk %60'ında seçilir, seçime hiç katılmayan son %40'ında sınanır. İç-örneklem optimizasyonunun ürettiği dört öneriden **ikisi reddedildi** — örneklem dışında belirgin biçimde daha kötü performans gösterdiler.

### Bilinen sınırlar

- Backtest yalnızca **teknik çekirdeği** test eder; haber ve analist verisi geriye dönük üretilemediği için bu, sahadaki 45/25/20/10 sisteminin kendisi değildir.
- Evren bugünün likit listesidir, borsadan çıkmış şirketler yoktur (hayatta kalma yanlılığı).
- Beş yıllık pencere bir boğa piyasasıdır; stop gibi sigortalar böyle bir örneklemde sadece maliyet gibi görünür.
- Kayma (slippage) modellenmemiştir; maliyet sabit yüzdedir.
- Canlı takip haftalarla ölçülür, yıllarla değil. Gerçekçi bir Sharpe'ta beceriyi şanstan ayırmak **yıllar** alır.

---

## Sayfalar

Her iki piyasa da aynı yedi sayfalık şekle sahiptir. ABD sayfaları kopya değil, piyasaya göre parametrelenmiş aynı fonksiyonlardır.

**Kokpit** (varsayılan açılış — piyasa rejimi, makro şerit, günün sinyalleri, sistem durumu) · **Portföy Yöneticisi** · **Hisse Listesi** · **Hisse Analizi** · **Portfolyum** · **Backtest** · **Sinyal Takip**

İki tema (aydınlık "Gazete", karanlık "Obsidyen"). Piyasa `?piyasa=US`, tema `?tema=` ile derin bağlanabilir.

**Sinyal Takip**, sistemin ürettiği her sinyali kaydeder ve 1/3/7/14/30 gün sonraki getirisini ölçer — backtest değil, canlı üretim takibi.

---

## Otomasyon

`daily_robot.py` GitHub Actions üzerinde hafta içi iki kez çalışır — BIST kapanışından sonra ve ABD kapanışından sonra; çünkü ABD pozisyonlarına seans ortasında karar vermek yarım günün fiyatını kullanmak olurdu. Her koşuda:

1. Her aktif portföyün NAV'ını günceller (piyasa başına)
2. Haftalık gölge ve kontrol kohortlarını tazeler
3. Günlük portföy yönetimini çalıştırır
4. Kural denetimini çalıştırır
5. Risk alarmlarını kontrol eder
6. Piyasa rejimini kaydeder
7. Aylık kalibrasyon ve enflasyon verisi hatırlatmalarını yönetir

---

## Teknoloji

| Katman | Teknoloji |
|---|---|
| Arayüz | Streamlit + Plotly |
| Fiyat verisi | yfinance |
| Evren & analist hedefleri | TradingView screener |
| Haber | feedparser (20+ RSS kaynağı), KAP bildirimleri için borsapy |
| Duyarlılık | Türkçe BERT (opsiyonel), anahtar kelime yedeği |
| Depolama | Turso (HTTP üzerinden libSQL), yerel SQLite yedeği |
| Otomasyon | GitHub Actions |

---

## Kurulum

```bash
git clone https://github.com/Akif3445/bist-analyzer.git
cd bist-analyzer
pip install -r requirements.txt
streamlit run bist_analyzer.py
```

Bağımlılıklar tedarik zinciri önlemi olarak tam sürüme sabitlenmiştir — CI, veritabanı yazma anahtarı ortamdayken kurulum yapar; sabitlenmemiş, ele geçirilmiş bir sürüm o anahtarı sızdırabilir. Sürümleri otomatik değil, bilinçli güncelleyin.

Opsiyonel `.env`:

```
ANTHROPIC_API_KEY=...      # yapay zekâ Türkçe özet
TURSO_DATABASE_URL=...     # bulut kalıcılık; yoksa yerel SQLite'a düşer
TURSO_AUTH_TOKEN=...
APP_EDIT_KEY=...           # yazma koruması (aşağıya bkz.)
```

Opsiyonel Türkçe BERT duyarlılığı (~2 GB; Streamlit Cloud ücretsiz katmanı RAM kısıtlı olduğu için requirements'ta değil):

```bash
pip install transformers torch
```

### Araştırma araçları

```bash
python weight_calibration.py build-tech      # point-in-time panel kur
python weight_calibration.py analyze         # IC analizi + önerilen ağırlıklar
python pipeline_backtest.py run 5            # tam-hat simülasyon
python pipeline_backtest.py sweep 5          # yönetim kuralı taraması
python pipeline_backtest.py validate 8       # örneklem-dışı doğrulama
python tools/pmdb_admin.py enag-list         # veritabanı bakımı
```

---

## Güvenlik

Halka açık demo, veritabanı yazma kimlik bilgileriyle çalışır ve Streamlit'in yerleşik kimlik doğrulaması yoktur. `APP_EDIT_KEY` tanımlıysa tüm yazma işlemleri — portföy kaydetme, enflasyon tablosu düzenleme, kural parametresi değiştirme — kenar çubuğundan kilit açmayı gerektirir; okumalar herkese açık kalır. Anahtar tanımlı değilse (yerel geliştirme) yazmalar korumasızdır ve davranış değişmez.

HTML olarak çizilen dış RSS içeriği escape edilir, link şemaları kontrol edilir. SQL baştan sona parametrelidir. Günlük robot en az yetkiyle çalışır (`contents: read`).

**Halka açık demoya gerçek kişisel pozisyonlarınızı girmeyin** — yazma koruması kirlenmeyi engeller, görünmeyi değil.

---

## Sorumluluk reddi

Yalnızca bilgilendirme ve eğitim amaçlıdır. Yatırım tavsiyesi değildir ve buradaki hiçbir şey herhangi bir menkul kıymeti alma veya satma önerisi değildir. Tüm yatırım kararları risk taşır, geçmiş performans gelecek sonuçları garanti etmez ve bu sistemin canlı kanıtı şu an **haftalarla** ölçülmektedir — beceriyi kanıtlamak için fazlasıyla kısa bir süre. Kendi araştırmanızı yapın.

---

## Lisans

Tescilli. Tüm hakları saklıdır.
