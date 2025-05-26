import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('bank-additional.xls', sep='\t')

# Fix numeric columns
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                   'euribor3m', 'nr.employed']

for col in numeric_columns:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.') if isinstance(df[col].iloc[0], str) else df[col]
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Separate approved and rejected customers
approved = df[df['y'] == 'yes']
rejected = df[df['y'] == 'no']

print('=== ONAY ALAN MÜŞTERİLERİN ÖZELLİKLERİ ===')
print(f'Toplam onay alan müşteri sayısı: {len(approved)} ({len(approved)/len(df)*100:.1f}%)')
print()

print('1. GÖRÜŞME SÜRESİ (Duration):')
print(f'   - Onay alanlar ortalama: {approved["duration"].mean():.0f} saniye')
print(f'   - Onay almayanlar ortalama: {rejected["duration"].mean():.0f} saniye')
print(f'   - Onay alanların %75\'i: {approved["duration"].quantile(0.75):.0f} saniye üzeri')
print(f'   - 300+ saniye görüşme yapanların onay oranı: {len(df[(df["duration"] >= 300) & (df["y"] == "yes")]) / len(df[df["duration"] >= 300]) * 100:.1f}%')
print()

print('2. ÖNCEKİ KAMPANYA SONUCU (poutcome):')
poutcome_success = len(approved[approved['poutcome'] == 'success'])
poutcome_total_success = len(df[df['poutcome'] == 'success'])
if poutcome_total_success > 0:
    print(f'   - Önceki kampanyada başarılı olanların onay oranı: {poutcome_success/poutcome_total_success*100:.1f}%')
print(f'   - Onay alanların {len(approved[approved["poutcome"] == "success"])/len(approved)*100:.1f}%\'i önceki kampanyada başarılı')
print()

print('3. SON GÖRÜŞMEDEN GEÇİRİLEN GÜN (pdays):')
print(f'   - Onay alanlar ortalama pdays: {approved["pdays"].mean():.0f}')
print(f'   - Daha önce hiç aranmayanlar (999): {len(approved[approved["pdays"] == 999])/len(approved)*100:.1f}%')
recent_contacted = df[(df["pdays"] < 30) & (df["pdays"] != 999)]
if len(recent_contacted) > 0:
    print(f'   - 30 gün içinde aranmış olanların onay oranı: {len(recent_contacted[recent_contacted["y"] == "yes"]) / len(recent_contacted) * 100:.1f}%')
print()

print('4. ÖNCEKİ KAMPANYA SAYISI (previous):')
print(f'   - Onay alanlar ortalama: {approved["previous"].mean():.1f}')
previous_contacted = df[df["previous"] > 0]
if len(previous_contacted) > 0:
    print(f'   - Daha önce 1+ kez aranmış olanların onay oranı: {len(previous_contacted[previous_contacted["y"] == "yes"]) / len(previous_contacted) * 100:.1f}%')
print()

print('5. YAŞ DAĞILIMI:')
print(f'   - Onay alanlar ortalama yaş: {approved["age"].mean():.1f}')
print(f'   - En çok onay alan yaş grubu: {approved["age"].mode().iloc[0]} yaş')
print()

print('6. MESLEKİ DAĞILIM:')
job_approval = df.groupby('job')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100).sort_values(ascending=False)
print('   En yüksek onay oranına sahip meslekler:')
for job, rate in job_approval.head(5).items():
    count = len(approved[approved['job'] == job])
    print(f'   - {job}: %{rate:.1f} onay oranı ({count} kişi)')
print()

print('7. EĞİTİM DÜZEYİ:')
edu_approval = df.groupby('education')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100).sort_values(ascending=False)
print('   En yüksek onay oranına sahip eğitim seviyeleri:')
for edu, rate in edu_approval.head().items():
    count = len(approved[approved['education'] == edu])
    print(f'   - {edu}: %{rate:.1f} onay oranı ({count} kişi)')
print()

print('8. İLETİŞİM TİPİ:')
contact_approval = df.groupby('contact')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100).sort_values(ascending=False)
for contact, rate in contact_approval.items():
    count = len(approved[approved['contact'] == contact])
    print(f'   - {contact}: %{rate:.1f} onay oranı ({count} kişi)')
print()

print('9. MEDENİ DURUM:')
marital_approval = df.groupby('marital')['y'].apply(lambda x: (x == 'yes').sum() / len(x) * 100).sort_values(ascending=False)
for marital, rate in marital_approval.items():
    count = len(approved[approved['marital'] == marital])
    print(f'   - {marital}: %{rate:.1f} onay oranı ({count} kişi)')
print()

print('=== KESİNLİKLE ONAY ALMASI GEREKEN PROFIL ===')
print('Aşağıdaki özelliklere sahip müşteriler yüksek onay olasılığına sahiptir:')
print()
print('✓ Görüşme süresi 400+ saniye (ilgi düzeyi yüksek)')
print('✓ Önceki kampanyada başarılı sonuç almış (poutcome = success)')
print('✓ Son 30 gün içinde aranmış ve olumlu yanıt vermiş')
print('✓ Daha önce 1+ kez kampanyaya katılmış')
print('✓ Cellular ile iletişim kurulan müşteriler')
print('✓ Retired, student, admin, management mesleklerinden')
print('✓ University degree veya professional course eğitim seviyesi')
print('✓ 25-45 yaş arası (en aktif grup)')

# High probability combinations
print()
print('=== YÜKSEK OLASILIK KOMBİNASYONLARI ===')

# Combination 1: Previous success + long duration
combo1 = df[(df['poutcome'] == 'success') & (df['duration'] >= 300)]
if len(combo1) > 0:
    success_rate1 = len(combo1[combo1['y'] == 'yes']) / len(combo1) * 100
    print(f'1. Önceki başarı + 300+ saniye görüşme: %{success_rate1:.1f} onay oranı ({len(combo1)} kişi)')

# Combination 2: Student + long duration
combo2 = df[(df['job'] == 'student') & (df['duration'] >= 200)]
if len(combo2) > 0:
    success_rate2 = len(combo2[combo2['y'] == 'yes']) / len(combo2) * 100
    print(f'2. Öğrenci + 200+ saniye görüşme: %{success_rate2:.1f} onay oranı ({len(combo2)} kişi)')

# Combination 3: Retired + cellular contact
combo3 = df[(df['job'] == 'retired') & (df['contact'] == 'cellular')]
if len(combo3) > 0:
    success_rate3 = len(combo3[combo3['y'] == 'yes']) / len(combo3) * 100
    print(f'3. Emekli + cellular iletişim: %{success_rate3:.1f} onay oranı ({len(combo3)} kişi)')

# Combination 4: University degree + previous contact
combo4 = df[(df['education'] == 'university.degree') & (df['previous'] > 0)]
if len(combo4) > 0:
    success_rate4 = len(combo4[combo4['y'] == 'yes']) / len(combo4) * 100
    print(f'4. Üniversite mezunu + önceki iletişim: %{success_rate4:.1f} onay oranı ({len(combo4)} kişi)') 