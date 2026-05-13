# CNN GPU Hızlandırmalı Güneş Paneli Kar Tespiti

Bu depo, güneş panellerindeki **kar tespiti** için GPU hızlandırmalı derin öğrenme sisteminin uygulama ve deneysel çerçevesini içermektedir. CNN mimarileri (VGG-19 ve ResNet-50) kullanılarak **CPU çok iş parçacıklı eğitim ile GPU hızlandırmalı eğitim** karşılaştırılmıştır. Deneyler TRUBA ulusal yüksek başarımlı hesaplama altyapısında gerçekleştirilmiştir.

---

## Amaç

- CPU çok iş parçacıklı eğitim ile GPU eğitimi performans analizi
- Amdahl Yasasının gerçek HPC sistemlerinde deneysel doğrulanması
- VGG-19 ve ResNet-50 mimarilerinin paralel ölçekleme davranışının karşılaştırılması
- Uygulama alanı olarak güneş paneli kar tespiti

---

## Modeller

- VGG-19 (ImageNet ön eğitimli)
- ResNet-50 (ImageNet ön eğitimli)

Girdi:

- 192×192 RGB görüntüler

Çıktı:

- 3 sınıf:
  - `all_snow` (tam karlı)
  - `no_snow` (karsız)
  - `partial` (kısmen karlı)

---

## Donanım ve Yazılım Ortamı

### Donanım

| Bileşen                  | Özellik                  |
| ------------------------- | ------------------------- |
| CPU                       | Intel Xeon Platinum 8480+ |
| Toplam İş Parçacığı | 112                       |
| GPU                       | NVIDIA V100 SXM2 16GB     |
| CUDA Çekirdeği          | 5120                      |
| Altyapı                  | TRUBA YBH Kümesi         |

### Yazılım

| Bileşen         | Sürüm      |
| ---------------- | ------------ |
| Python           | 3.12.2       |
| PyTorch          | 2.1.0        |
| CUDA             | 12.0         |
| Paralelleştirme | OpenMP / MKL |

---

## Deneyler

### CPU Deneyleri

- Tek iş parçacıklı temel referans
- Çok iş parçacıklı ölçekleme: {1, 2, 4, 7, 14, 28, 56, 112} iş parçacığı

### GPU Deneyleri

- V100 hızlandırma testleri
- Toplu iş boyutuna dayalı eğitim karşılaştırması

---

## Temel Bulgular

- VGG-19 CPU hızlanması: **28×'e kadar** (56 iş parçacığında, f≈%1.8)
- ResNet-50 CPU hızlanması: **~1.56×** (f≈%63.4)
- GPU hızlanması (1 iş parçacıklı CPU'ya kıyasla):
  - ResNet-50: **163×**
  - VGG-19: **1067×**
- GPU hızlandırması mimari bağımsızlığını korumaktadır
- 18 donanım yapılandırmasının tamamında sınıflandırma kalitesi değişmez kalmıştır (doğruluk ≥ 0.997, F1 ≥ 0.995)
- 112 iş parçacığında NUMA etkisiyle VGG-19 zirve hızlanmasının %36'sını kaybetmiştir; optimum yapılandırma 56 iş parçacığıdır

---

## Depo Yapısı

```text
.
├── src/               # Eğitim scriptleri
├── slurm/             # HPC iş kuyruğu betikleri
│   ├── cpu/
│   └── gpu/
├── results/           # Özet sonuçlar (CSV, JSON)
│   └── plots/         # Makale ve tez şekilleri
├── logs/              # SLURM çıktı logları
├── docs/              # Metodoloji ve matematik
├── README.md          # İngilizce açıklama
└── README_TR.md       # Türkçe açıklama
```

---

## Metodoloji

- Veri artırma: yatay çevirme, döndürme, renk bozma
- Ağırlıklı çapraz entropi kaybı (sınıf dengesizliği için)
- Adam optimizasyon algoritması (lr=1e-4, weight decay=1e-4)
- ReduceLROnPlateau öğrenme hızı zamanlayıcısı (patience=5, factor=0.5)
- Erken durdurma (patience=7)
- Katmanlı örnekleme ile 70/15/15 veri bölünmesi

---

## Notlar

- Bu depo ham veri kümesi görüntülerini içermemektedir
- Yalnızca işlenmiş sonuçlar ve loglar saklanmaktadır
- YBH ortamlarında tekrarlanabilirlik gözetilerek tasarlanmıştır

---

## Atıf

Bu çalışmayı akademik araştırmalarınızda kullanmanız durumunda aşağıdaki kaynağa atıf yapmanız rica olunur:

Turkan, A., Hangun, B. and Eyecioglu, O. (2026). HPC-Accelerated CNN Training for Solar Panel Snow Detection: A Comparative Analysis of Multi-threaded CPU and GPU Performance with VGG-19 and ResNet-50. *International Journal of Smart Grid*. (in press)

İngilizce dokümantasyon için: [README.md](README.md)
