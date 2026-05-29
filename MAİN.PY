PYTHON KOD 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 1. CSV dosyasini oku
df = pd.read_csv("countries of the world.csv")

# 2. Ulke isimlerindeki bosluklari temizle
df["Country"] = df["Country"].str.strip()

# 3. Kullanilacak ulkeler
selected_countries = ["Turkey", "Algeria", "Germany", "China", "United States", "Japan"]
df_selected = df[df["Country"].isin(selected_countries)].copy()

# 4. Kullanilacak sutunlar
columns = [
    "Country",
    "Population",
    "GDP ($ per capita)",
    "Agriculture",
    "Industry",
    "Service"
]

df_selected = df_selected[columns]

# 5. Sayisal sutunlari duzenle
numeric_columns = [
    "Population",
    "GDP ($ per capita)",
    "Agriculture",
    "Industry",
    "Service"
]

for col in numeric_columns:
    df_selected[col] = (
        df_selected[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")

# 6. Eksik verileri sil
df_selected = df_selected.dropna()

print("Secilen veri tablosu:")
print(df_selected)

# 7. Matris olusturma
X = df_selected[numeric_columns]

# 8. Verileri standartlastirma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9. PCA uygulama
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Country"] = df_selected["Country"].values

print("\nPCA Sonuclari:")
print(pca_df)

# 10. PCA performans degerleri
explained_variance = pca.explained_variance_ratio_

print("\nPCA Aciklanan Varyans Oranlari:")
print("PC1:", explained_variance[0])
print("PC2:", explained_variance[1])
print("Toplam:", explained_variance.sum())

# 11. Cosine Similarity hesaplama
similarity_matrix = cosine_similarity(X_scaled)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df_selected["Country"],
    columns=df_selected["Country"]
)

print("\nUlkeler Arasi Benzerlik Matrisi:")
print(similarity_df)

# 12. Sonuclari dosyaya kaydet
pca_df.to_csv("pca_results.csv", index=False, encoding="utf-8")
similarity_df.to_csv("similarity_results.csv", encoding="utf-8")

# 13. PCA grafigi
plt.figure(figsize=(8, 6))
plt.scatter(pca_df["PC1"], pca_df["PC2"])

for i, country in enumerate(pca_df["Country"]):
    plt.text(pca_df["PC1"][i], pca_df["PC2"][i], country)

plt.title("PCA ile Ulkelerin Ekonomik Benzerlik Analizi")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

# Grafigi kaydet
plt.savefig("pca_grafigi.png", dpi=300, bbox_inches="tight")

plt.show()

print("\nDosyalar olusturuldu:")
print("- pca_results.csv")
print("- similarity_results.csv")
print("- pca_grafigi.png")