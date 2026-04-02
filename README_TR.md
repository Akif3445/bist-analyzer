# BIST Akilli Yatirim Asistani

Borsa Istanbul (BIST) ve ABD piyasalari icin kapsamli bir hisse analiz platformu. Sistem; teknik analiz, haber duyarlilik analizi ve degerleme metriklerini birlestirerek 0-100 araliginda bir bileşik skor uzerinden yatirim sinyalleri uretir.

Canli demo: [bist-analyzer.streamlit.app](https://bist-analyzer.streamlit.app)

---

## Genel Bakis

Platform, birden fazla veri kaynagini es zamanli isleyerek otomatik hisse analizi yapar. Anlik fiyat verisi ceker, 20'den fazla Turkce ve Ingilizce kaynaktan haber duyarlilik analizi yapar ve teknik gostergeleri hesaplayarak her hisse icin birlesik bir al/sat sinyali uretir.

Sistem 80'den fazla BIST hissesini ve 50'den fazla ABD hissesini kapsar. Tum analizler Streamlit uzerinden istemci tarafinda calisir, yerel onbellekleme ve portfoy kaliciligi icin SQLite kullanilir.

---

## Sistem Mimarisi

Uygulama iki ana modulden olusur:

**bist_analyzer.py** — Ana uygulama (~8500 satir). Tum skorlama motorlari, sayfa goruntuleyicileri, veritabani islemleri, geriye donuk test mantigi ve Streamlit arayuzunu icerir.

**news_engine.py** — Haber duyarlilik motoru (~1600 satir). 20'den fazla kaynaktan RSS cekmeyi, hibrit duyarlilik siniflandirmasini (BERT + anahtar kelime tabanli), kaynak guvenilirlik agirliklandirmasini ve spekulasyon filtrelemeyi yonetir.

---

## Skorlama Modeli

Her hisse dort bilesenden hesaplanan 0 ile 100 arasinda bir bileşik skor alir:

| Bilesen | Agirlik | Kaynak |
|---------|---------|--------|
| Teknik Skor | %35 | SMA, RSI, MACD, Bollinger, Stochastic, OBV, ADX |
| Haber Duyarliligi | %35 | 20+ RSS kaynak, kaynak guvenilirligine gore agirlikli |
| Yukselis Potansiyeli | %20 | Analist hedef fiyati ile mevcut fiyat karsilastirmasi |
| Degerleme | %10 | F/K, PD/DD oranlari sektor ortalamarina gore |

Analist hedef fiyati bulunmadiginda agirlik Teknik ve Duyarlilik bilesenlerine yeniden dagitilir. Temel veri eksik oldugunda da ayni durum gecerlidir.

### Sinyal Esikleri

| Skor Araligi | Sinyal | Anlami |
|--------------|--------|--------|
| 72 - 100 | GUCLU AL | Birden fazla gosterge olumlu yonde hizali |
| 57 - 71 | AL | Gostergelerin cogunlugu olumlu |
| 43 - 56 | NOTR | Karisik sinyaller, net bir yon yok |
| 30 - 42 | SAT | Gostergelerin cogunlugu olumsuz |
| 0 - 29 | GUCLU SAT | Birden fazla gosterge olumsuz yonde hizali |

---

## Teknik Gostergeler

TechnicalEngine, OHLCV verisinden asagidaki gostergeleri hesaplar:

- **SMA 50/200**: Trend yonu, altin/olum caprazlama tespiti
- **RSI (14)**: Momentum, asiri alim/asiri satim seviyeleri
- **MACD (12, 26, 9)**: Sinyal cizgisi caprazlamalari, histogram sapmalari
- **Bollinger Bantlari (20, 2 sigma)**: Volatilite ve ortalamaya donus
- **Stochastic Osilatoru (14, 3)**: Momentum teyidi
- **OBV (Denge Hacmi)**: Hacim-fiyat sapmalari
- **ADX (14)**: Trend gucu filtresi (ADX > 25 oldugunda carpan uygulanir)
- **ATR (14)**: Risk hesaplamalari icin volatilite olcumu
- **52 Haftalik Pozisyon**: Mevcut fiyatin yillik araliga gore konumu

Her gosterge grubu toplam teknik skora bir alt skor katkisi yapar (maks 100). ADX dogrudan katkida bulunmak yerine carpan gorevi gorur — guclu trendler mevcut sinyalleri buyutur.

---

## Haber Duyarlilik Motoru

Duyarlilik motoru uc katmanli bir siniflandirma yaklasimi kullanir:

**Katman 1 — Guclu Kalip Gecersiz Kilma**: Kesin finansal olaylar icin regex kaliplari (orn. "rekor kar acikladi", "iflas basvurusu"). Bunlar diger tum katmanlari atlar.

**Katman 2 — BERT Modeli**: Mevcut oldugunda, Turkce dil anlayisi icin `savasy/bert-base-turkish-sentiment-cased` modelini kullanir. Guveni >= %70 olan tahminler dogrudan kabul edilir.

**Katman 3 — Agirlikli Anahtar Kelimeler**: Siddet agirliklari olan (1x normal, 2x guclu, 3x kritik) 120'den fazla olumlu ve 98'den fazla olumsuz finansal terim. BERT kullanilamaz veya belirsiz oldugunda buna geri doner.

### Kaynak Guvenilirligi

Haber kaynaklari uc katmana ayrilir:

| Katman | Agirlik | Kaynaklar |
|--------|---------|-----------|
| Katman 1 (3x) | En yuksek | Bloomberg HT, NTV Ekonomi, Haberturk |
| Katman 2 (2x) | Standart | Hurriyet, Dunya Gazetesi, Investing.com TR, CNN Turk |
| Katman 3 (1x) | Ek kaynak | Google News, Bing News, Yahoo Finance |

Nihai haber skorunu etkileyen ek faktorler:
- **Guncellik**: Son 2 gunun haberleri 1.5x agirlik alir, 7+ gunluk haberler 0.7x
- **Resmi terimler**: KAP bildirimleri, finansal raporlar ve kurumsal eylemler guven bonusu alir
- **Mukerrer tespit**: Birden fazla kaynaktan gelen ayni haber guveni artirir
- **Spekulasyon filtresi**: Sosyal medya dili, manipulasyon terimleri ve dogrulanmamis iddialar haric tutulur

---

## Geriye Donuk Test (Backtest)

BacktestEngine bes strateji modunu destekler:

| Mod | Aciklama |
|-----|----------|
| Swing | Kisa vadeli islemler, 5-20 gun tutma suresi |
| Trend | SMA caprazlamalari kullanarak yerlesik trendleri takip eder |
| Universal | Swing ve trend sinyallerini birlestiren dengeli yaklasim |
| Investor | Genis zarar-kes seviyeleriyle uzun vadeli pozisyonlar |
| Buy & Hold | Karsilastirma icin kiyaslama, aktif islem yok |

Geriye donuk testler ileriye bakma sapmasini onlemek icin zaman noktasi (PIT) verisi kullanir. Her test noktasi yalnizca o tarihsel anda mevcut olan verileri kullanir.

---

## Sinyal Takip

Sinyal Takip modulu, sistemin urettigi her al/sat sinyalini kaydeder ve ardindan fiyat hareketini izler. Sinyaller bes zaman diliminde takip edilir: 1 gun, 3 gun, 7 gun, 14 gun ve 30 gun.

Goruntulenebilen veriler:
- Gunluk, haftalik ve aylik sinyal goruntuleri
- Her sinyal icin donem bazli getiri yuzdeleri
- Genel dogruluk istatistikleri
- Sinyal dagilimi ve performans dokumleri

Hem BIST hem de ABD piyasasi sinyalleri bagimsiz olarak takip edilir.

---

## Zaman Makinesi

Zaman Makinesi modulu "bugunun stratejisini gecmiste uygulasaydik ne olurdu?" sorusunu yanitlayan tam tarihsel simulasyonlar calistirir. Alti portfoy stilini destekler:

- Agresif, Defansif, Momentum, Deger, Istikrarli ve Ozel

Her simulasyon, her tarihsel noktada skorlama modelini kullanarak hisse secer, ardindan gercekci giris/cikis mantigiyla portfoy performansini ileriye dogru takip eder.

---

## Sayfalar

### BIST Piyasasi (8 sayfa)
1. **Piyasa Ozeti** — Makro gostergeler (USD/TL, EUR/TL, Altin, BIST100) ve bugunun sinyal ozeti
2. **BIST Listesi** — 80+ hissenin skora gore sirali taramasi
3. **Hisse Analizi** — Tek bir hisse icin tum gostergelerle derin analiz
4. **Portfolyum** — Kar/zarar takipli kisisel portfoy
5. **Backtest** — Tarihsel strateji testi
6. **Sistem Portfolyleri** — Onceden olusturulmus akilli portfoyler
7. **Zaman Makinesi** — Tarihsel "ya olsaydi" simulasyonlari
8. **Sinyal Takip** — Sinyal dogruluk takipçisi

### ABD Piyasasi (7 sayfa)
1. **US Analiz** — ABD hisseleri icin tek hisse analizi
2. **US Backtest** — ABD hisseleri icin geriye donuk test
3. **US Stock List** — ABD hisse tarayicisi
4. **US Portfolios** — ABD akilli portfoyleri
5. **Portfolyum** — Ortak portfoy sayfasi
6. **Zaman Makinesi** — Ortak zaman makinesi
7. **US Sinyal Takip** — ABD sinyal takipçisi

---

## Teknoloji

| Bilesen | Teknoloji |
|---------|-----------|
| Arayuz | Streamlit |
| Veri | yfinance |
| Haberler | feedparser, requests (20+ RSS kaynak) |
| Duyarlilik | BERT (opsiyonel), anahtar kelime tabanli (varsayilan) |
| Veritabani | SQLite |
| Grafikler | Plotly |
| Yayin | Streamlit Cloud |

---

## Kurulum

```bash
git clone https://github.com/Akif3445/bist-analyzer.git
cd bist-analyzer
pip install -r requirements.txt
```

`.env` dosyasi olusturun:
```
ANTHROPIC_API_KEY=anahtariniz
```

Calistirma:
```bash
streamlit run bist_analyzer.py
```

BERT duyarlilik icin (opsiyonel, ~2GB ek alan gerektirir):
```bash
pip install transformers torch
```
BERT kurulu olmadiginda uygulama otomatik olarak anahtar kelime tabanli duyarliliga geri doner.

---

## Yapilandirma

`.streamlit/config.toml` dosyasi tema ve sunucu ayarlarini icerir. Uygulama varsayilan olarak koyu tema kullanir.

Gizli bilgiler su yollarla yapilandirilabilir:
1. `.env` dosyasi (yerel gelistirme)
2. Streamlit Cloud gizli bilgiler paneli (uretim)

Uygulama oncelikle `st.secrets` okur, bulamazsa `os.getenv` kullanir.

---

## Veri Kaynaklari

**Fiyat Verisi**: yfinance kutuphanesi araciligiyla Yahoo Finance. Tum BIST ve buyuk ABD borsasi hisselerini destekler.

**Haber Kaynaklari (Turkce)**: Bloomberg HT, NTV Ekonomi, Haberturk, Hurriyet, Milliyet, Sabah, Dunya Gazetesi, Investing.com TR, Finans Gundem, Ekonomim, Para Analiz, Foreks, CNN Turk Ekonomi, Google News TR

**Haber Kaynaklari (Ingilizce)**: Google News EN, Bing News, Yahoo Finance, Yahoo Finance RSS

---

## Sorumluluk Reddi

Bu yazilim yalnizca bilgilendirme ve egitim amaciyldir. Finansal tavsiye niteliginde degildir. Tum yatirim kararlari risk tasir. Gecmis performans gelecekteki sonuclari garanti etmez. Yatirim kararlari almadan once her zaman kendi arastirmanizi yapin.

---

## Lisans

Bu proje tescillidir. Tum haklari saklidir.
