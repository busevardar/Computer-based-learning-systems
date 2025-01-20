from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd

# Sürekli değişkenleri ölçekle
scaler = MinMaxScaler()
scaled_continuous = scaler.fit_transform(df[num_cols])

# Kategorik değişkenleri one-hot encode et
encoder = OneHotEncoder(sparse=False)
encoded_categorical = encoder.fit_transform(df[cat_cols])

# TALEP_NO'yu array'e dönüştür (ölçeklemeden)
talep_no_array = df['TALEP_NO'].values.reshape(-1, 1)

# Sürekli, kategorik, TARGET ve TALEP_NO verilerini birleştir
normalized_data = np.hstack([scaled_continuous, encoded_categorical, df['TARGET'].values.reshape(-1, 1), talep_no_array])

# Kolon isimlerini belirle (num_cols, one-hot encoded kategoriler, TARGET, ve TALEP_NO)
encoded_cat_columns = encoder.get_feature_names_out(cat_cols)  # Kategorik değişkenlerin one-hot sütun isimleri
columns = list(num_cols) + list(encoded_cat_columns) + ['TARGET', 'TALEP_NO']

# DataFrame'e dönüştür
df_normalized = pd.DataFrame(normalized_data, columns=columns)

# TALEP_NO sütununun tipini int olarak ayarla
df_normalized['TALEP_NO'] = df_normalized['TALEP_NO'].astype(int)

# Sonuç
print(df_normalized.head())


from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


from sklearn.model_selection import train_test_split

def prepare_data_by_group(df, group_column, columns, target_column="TARGET", test_size=0.2, val_size=0.2, random_state=42):
    """
    Prepares train, test, and validation datasets by splitting based on unique group values.

    Parameters:
    - df: The dataset (DataFrame)
    - group_column: The column name to group by (e.g., "KF_LOAN_NUM")
    - columns: List of column names for the features to be used in the model
    - target_column: The name of the target variable (default is "TARGET")
    - test_size: Proportion of groups to include in the test set (default is 0.2)
    - val_size: Proportion of groups to include in the validation set (default is 0.2)
    - random_state: Seed for reproducibility (default is 42)

    Returns:
    - X_train, y_train: Training data and target variable
    - X_valid, y_valid: Validation data and target variable
    - X_test, y_test: Test data and target variable
    - train_data, test_data, valid_data: Original train, test, and validation datasets
    """
    # Get unique group values
    unique_groups = df[group_column].unique()

    # Split unique groups into train and temporary (test + validation)
    train_groups, temp_groups = train_test_split(unique_groups, test_size=(test_size + val_size), random_state=random_state)

    # Calculate the proportion of test and validation groups within the temporary groups
    val_test_ratio = val_size / (test_size + val_size)
    valid_groups, test_groups = train_test_split(temp_groups, test_size=(1 - val_test_ratio), random_state=random_state)

    # Filter data based on the group splits
    train_data = df[df[group_column].isin(train_groups)]
    valid_data = df[df[group_column].isin(valid_groups)]
    test_data = df[df[group_column].isin(test_groups)]

    # Check for intersections between the splits
    if (set(train_data[group_column]) & set(valid_data[group_column])):
        raise ValueError("Train and validation groups overlap!")
    if (set(train_data[group_column]) & set(test_data[group_column])):
        raise ValueError("Train and test groups overlap!")
    if (set(valid_data[group_column]) & set(test_data[group_column])):
        raise ValueError("Validation and test groups overlap!")

    # Separate features and target variables
    X_train = train_data[columns]
    y_train = train_data[target_column]
    X_valid = valid_data[columns]
    y_valid = valid_data[target_column]
    X_test = test_data[columns]
    y_test = test_data[target_column]

    return X_train, y_train, X_valid, y_valid, X_test, y_test, train_data, test_data, valid_data


def check_intersection(train_data, valid_data, test_data, group_column="TALEP_NO"):
    """
    Verisetlerinin kesişiminin boş küme olup olmadığını kontrol eder.

    Parameters:
    - train_data: Eğitim veri seti (DataFrame)
    - valid_data: Doğrulama veri seti (DataFrame)
    - test_data: Test veri seti (DataFrame)
    - group_column: Grup sütunu (örneğin, KF_LOAN_NUM)

    Returns:
    - Kesişim bilgisi ve gerekli çıktılar
    """
    # Grup değerlerini set olarak al
    train_groups = set(train_data[group_column])
    valid_groups = set(valid_data[group_column])
    test_groups = set(test_data[group_column])

    # Kesişimleri kontrol et
    train_valid_intersection = train_groups.intersection(valid_groups)
    train_test_intersection = train_groups.intersection(test_groups)
    valid_test_intersection = valid_groups.intersection(test_groups)

    # Sonuçları yazdır
    if train_valid_intersection:
        print("Eğitim ve doğrulama veri setleri kesişiyor:", train_valid_intersection)
    else:
        print("Eğitim ve doğrulama veri setleri kesişmiyor.")
    
    if train_test_intersection:
        print("Eğitim ve test veri setleri kesişiyor:", train_test_intersection)
    else:
        print("Eğitim ve test veri setleri kesişmiyor.")
    
    if valid_test_intersection:
        print("Doğrulama ve test veri setleri kesişiyor:", valid_test_intersection)
    else:
        print("Doğrulama ve test veri setleri kesişmiyor.")

check_intersection(train_data, valid_data, test_data, group_column="TALEP_NO")


def train_and_evaluate_model(model, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Train and evaluate the model with metrics and visualizations.

    Parameters:
    - model: The machine learning model to train
    - model_name: The name of the model (e.g., 'Logistic Regression', 'Random Forest', 'LightGBM')
    - X_train: Training data features
    - y_train: Training data target
    - X_valid: Validation data features
    - y_valid: Validation data target
    - X_test: Test data features
    - y_test: Test data target
    """
    # Handle missing values if model is Logistic Regression or Random Forest
    if model_name in ['Logistic Regression', 'Random Forest']:
        imputer = SimpleImputer(strategy='constant', fill_value=-1)
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    # Performance Metrics Calculation
    def print_metrics(y_true, y_pred, dataset_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"\n{model_name} - {dataset_name} Metrics:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return accuracy, precision, recall, f1

    # Performance Results
    train_metrics = print_metrics(y_train, y_train_pred, "Train")
    valid_metrics = print_metrics(y_valid, y_valid_pred, "Validation")
    test_metrics = print_metrics(y_test, y_test_pred, "Test")

    # Custom Threshold Predictions and Metrics
    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        threshold = 0.4  # Custom threshold
        y_train_pred_custom = (y_train_proba >= threshold).astype(int)
        y_valid_pred_custom = (y_valid_proba >= threshold).astype(int)
        y_test_pred_custom = (y_test_proba >= threshold).astype(int)

        def print_custom_threshold_metrics(y_true, y_pred, dataset_name):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            print(f"\n{model_name} - {dataset_name} Metrics (Custom Threshold {threshold}):")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
            return accuracy, precision, recall, f1

        print_custom_threshold_metrics(y_train, y_train_pred_custom, "Train")
        print_custom_threshold_metrics(y_valid, y_valid_pred_custom, "Validation")
        print_custom_threshold_metrics(y_test, y_test_pred_custom, "Test")

    # Confusion Matrix Plotting
    def plot_confusion_matrix(y_true, y_pred, dataset_name):
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(set(y_true))
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f"{model_name} - {dataset_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # Train/Validation/Test Confusion Matrices
    plot_confusion_matrix(y_train, y_train_pred, "Train")
    plot_confusion_matrix(y_valid, y_valid_pred, "Validation")
    plot_confusion_matrix(y_test, y_test_pred, "Test")

    # Feature Importance Visualization
    def plot_feature_importance(model, X_train, model_name):
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            feature_names = X_train.columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)

            # Top 50 Features
            top_50_importance_df = importance_df.head(50)
            plt.figure(figsize=(16, 12))
            ax = sns.barplot(x='Importance', y='Feature', data=top_50_importance_df, palette='viridis')

            for p in ax.patches:
                ax.annotate(f'{p.get_width():.4f}', (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2),
                            xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=12, color='black')

            plt.title(f'{model_name} - Top 50 Feature Importance', fontsize=16)
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()

            # Print remaining features
            remaining_features = importance_df.iloc[50:]
            print("\nRemaining Features and their Importance:")
            print(remaining_features)
        else:
            print(f"{model_name} does not provide feature importance")

    # Plot feature importance
    plot_feature_importance(model, X_train, model_name)


logistic_model = LogisticRegression(random_state=42)

# Fonksiyonu çağırma
train_and_evaluate_model(
    model=logistic_model,
    model_name="Logistic Regression",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test
)


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

# -----------------------------------
# GAN Parametreleri
# -----------------------------------
latent_dim = 16  # Latent space boyutu
epochs = 500  # Eğitim için epoch sayısı
batch_size = 64  # Mini-batch boyutu
learning_rate = 0.0002  # Öğrenme oranı
beta_1 = 0.5  # Adam optimizer için beta_1

# -----------------------------------
# GAN Modelini Tanımlama
# -----------------------------------

# Generator
def build_generator(input_dim, output_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(128),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(output_dim, activation='sigmoid')  # Çıkış özellik boyutuna eşit
    ])
    return model

# Discriminator
def build_discriminator(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(64),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')  # Gerçek/Sahte ayrımı
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate, beta_1), metrics=['accuracy'])
    return model

# GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_data = generator(gan_input)
    validity = discriminator(generated_data)
    gan = Model(gan_input, validity)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate, beta_1))
    return gan

# -----------------------------------
# Veriyi Hazırlama
# -----------------------------------
# TALEP_NO ve TARGET'ı çıkar
X = df_normalized.drop(columns=['TARGET', 'KF_LOAN_NUM']).values
target_values = df_normalized['TARGET'].values
talep_no_values = df_normalized['KF_LOAN_NUM'].values

# Özellik sayısını alın
input_dim = X.shape[1]

# -----------------------------------
# GAN Modellerini Oluşturma
# -----------------------------------
generator = build_generator(latent_dim, input_dim)
discriminator = build_discriminator(input_dim)
gan = build_gan(generator, discriminator)

# -----------------------------------
# GAN Eğitim Fonksiyonu
# -----------------------------------
for epoch in range(epochs):
    # Gerçek veriden rastgele bir batch seç
    idx = np.random.randint(0, X.shape[0], batch_size)
    real_data = X[idx]

    # Rastgele latent vektörler oluştur
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_data = generator.predict(noise)

    # Discriminator için gerçek ve sahte etiketler oluştur
    real_labels = np.ones((batch_size, 1))  # Gerçek veriler için etiketler
    fake_labels = np.zeros((batch_size, 1))  # Sahte veriler için etiketler

    # Discriminator'ı eğit
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_data, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Generator'ı eğit
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    # Her 100 epoch'ta bir kayıp değerlerini yazdır
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - D Loss: {d_loss[0]}, D Acc: {d_loss[1]} - G Loss: {g_loss}")

# -----------------------------------
# Sentetik Veri Üretimi
# -----------------------------------
num_samples = len(X)
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_X = generator.predict(noise)

# Sentetik veriyi DataFrame'e dönüştür
synthetic_df_gan_2 = pd.DataFrame(synthetic_X, columns=df_normalized.drop(columns=['TARGET', 'KF_LOAN_NUM']).columns)

# Orijinal TARGET ve TALEP_NO dağılımlarını kontrol et
target_distribution = df_normalized['TARGET'].value_counts(normalize=True)
talep_no_distribution = df_normalized['KF_LOAN_NUM'].value_counts(normalize=True)

# TARGET ve TALEP_NO'yu sentetik veriye ekle
synthetic_targets = np.random.choice(
    [0, 1], size=num_samples, p=[target_distribution[0.0], target_distribution[1.0]]
)
synthetic_talep_no = np.random.choice(
    df_normalized['TALEP_NO'].unique(), size=num_samples, replace=True
)

synthetic_df_gan_2['TARGET'] = synthetic_targets
synthetic_df_gan_2['TALEP_NO'] = synthetic_talep_no



# -----------------------------------
# Sonuçları İnceleme
# -----------------------------------
print("Sentetik TARGET Dağılımı:")
print(synthetic_df_gan_2['TARGET'].value_counts(normalize=True))

print("\nSentetik TALEP_NO Dağılımı:")
print(synthetic_df_gan_2['TALEP_NO'].value_counts(normalize=True))




X_train, y_train, X_valid, y_valid, X_test, y_test, train_data, test_data, valid_data = prepare_data_by_group(
    df=synthetic_df_gan_2,
    group_column="TALEP_NO",
    columns=columns,
    target_column="TARGET",
    test_size=0.2,
    val_size=0.2,
    random_state=42
)


logistic_model = LogisticRegression(random_state=42)

# Fonksiyonu çağırma
train_and_evaluate_model(
    model=logistic_model,
    model_name="Logistic Regression",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test
)


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam

# -----------------------------------
# CGAN Parametreleri
# -----------------------------------
latent_dim = 16  # Latent space boyutu
epochs = 500  # Eğitim için epoch sayısı
batch_size = 64  # Mini-batch boyutu
learning_rate = 0.0002  # Öğrenme oranı
beta_1 = 0.5  # Adam optimizer için beta_1

# -----------------------------------
# CGAN Modelini Tanımlama
# -----------------------------------

# Generator
def build_generator(latent_dim, feature_dim):
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    combined_input = Concatenate()([noise, label])
    hidden = Dense(64, activation='relu')(combined_input)
    hidden = BatchNormalization(momentum=0.8)(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    hidden = BatchNormalization(momentum=0.8)(hidden)
    output = Dense(feature_dim, activation='sigmoid')(hidden)
    return Model([noise, label], output)

# Discriminator
def build_discriminator(feature_dim):
    data = Input(shape=(feature_dim,))
    label = Input(shape=(1,))
    combined_input = Concatenate()([data, label])
    hidden = Dense(128, activation='relu')(combined_input)
    hidden = Dense(64, activation='relu')(hidden)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model([data, label], output)
    model.compile(optimizer=Adam(learning_rate, beta_1), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# CGAN
def build_cgan(generator, discriminator):
    discriminator.trainable = False
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    generated_data = generator([noise, label])
    validity = discriminator([generated_data, label])
    cgan = Model([noise, label], validity)
    cgan.compile(optimizer=Adam(learning_rate, beta_1), loss='binary_crossentropy')
    return cgan

# -----------------------------------
# Veriyi Hazırlama
# -----------------------------------
# TARGET ve TALEP_NO'yu çıkar
X = df_normalized.drop(columns=['TARGET', 'KF_LOAN_NUM']).values
target_values = df_normalized['TARGET'].values
num_samples, feature_dim = X.shape

# -----------------------------------
# CGAN Modellerini Oluşturma
# -----------------------------------
generator = build_generator(latent_dim, feature_dim)
discriminator = build_discriminator(feature_dim)
cgan = build_cgan(generator, discriminator)

# -----------------------------------
# CGAN Eğitim Fonksiyonu
# -----------------------------------
for epoch in range(epochs):
    # Gerçek veriden rastgele bir batch seç
    idx = np.random.randint(0, X.shape[0], batch_size)
    real_data = X[idx]
    real_labels = target_values[idx]

    # Rastgele latent vektörler oluştur ve sahte veri üret
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_labels = np.random.choice([0, 1], batch_size)
    generated_data = generator.predict([noise, fake_labels])

    # Discriminator için gerçek ve sahte veriyi birleştir
    data = np.vstack([real_data, generated_data])
    labels = np.hstack([real_labels, fake_labels])
    validity = np.hstack([np.ones(batch_size), np.zeros(batch_size)])

    # Discriminator'ı eğit
    d_loss = discriminator.train_on_batch([data, labels], validity)

    # Generator'ı eğit
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    random_labels = np.random.choice([0, 1], batch_size)
    g_loss = cgan.train_on_batch([noise, random_labels], np.ones(batch_size))

    # Her 100 epoch'ta bir kayıp değerlerini yazdır
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - D Loss: {d_loss[0]}, D Acc: {d_loss[1]} - G Loss: {g_loss}")

# -----------------------------------
# Sentetik Veri Üretimi
# -----------------------------------
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_labels = np.random.choice([0, 1], num_samples)
synthetic_data = generator.predict([noise, synthetic_labels])

# Sentetik veriyi DataFrame'e dönüştür
synthetic_df_cgan = pd.DataFrame(synthetic_data, columns=df_normalized.drop(columns=['TARGET', 'KF_LOAN_NUM']).columns)
synthetic_df_cgan['TARGET'] = synthetic_labels

# Orijinal TALEP_NO dağılımını kontrol et ve ekle
synthetic_talep_no = np.random.choice(df_normalized['KF_LOAN_NUM'].unique(), size=num_samples, replace=True)
synthetic_df_cgan['KF_LOAN_NUM'] = synthetic_talep_no

# -----------------------------------
# Sonuçları İnceleme
# -----------------------------------
print("Sentetik TARGET Dağılımı:")
print(synthetic_df_cgan['TARGET'].value_counts(normalize=True))

print("\nSentetik TALEP_NO Dağılımı:")
print(synthetic_df_cgan['KF_LOAN_NUM'].value_counts(normalize=True))



X_train, y_train, X_valid, y_valid, X_test, y_test, train_data, test_data, valid_data = prepare_data_by_group(
    df=synthetic_df_cgan,
    group_column="KF_LOAN_NUM",
    columns=columns,
    target_column="TARGET",
    test_size=0.2,
    val_size=0.2,
    random_state=42
)


logistic_model = LogisticRegression(random_state=42)

# Fonksiyonu çağırma
train_and_evaluate_model(
    model=logistic_model,
    model_name="Logistic Regression",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test
)


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam

# -----------------------------------
# CGAN Parametreleri
# -----------------------------------
latent_dim = 32  # Latent space boyutu
epochs = 500  # Eğitim için epoch sayısı
batch_size = 64  # Mini-batch boyutu
learning_rate = 0.0002  # Öğrenme oranı
beta_1 = 0.5  # Adam optimizer için beta_1

# -----------------------------------
# CGAN Modelini Tanımlama
# -----------------------------------

# Generator
def build_generator(latent_dim, feature_dim):
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    combined_input = Concatenate()([noise, label])
    hidden = Dense(64, activation='relu')(combined_input)
    hidden = BatchNormalization(momentum=0.8)(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    hidden = BatchNormalization(momentum=0.8)(hidden)
    output = Dense(feature_dim, activation='sigmoid')(hidden)
    return Model([noise, label], output)

# Discriminator
def build_discriminator(feature_dim):
    data = Input(shape=(feature_dim,))
    label = Input(shape=(1,))
    combined_input = Concatenate()([data, label])
    hidden = Dense(128, activation='relu')(combined_input)
    hidden = Dense(64, activation='relu')(hidden)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model([data, label], output)
    model.compile(optimizer=Adam(learning_rate, beta_1), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# CGAN
def build_cgan(generator, discriminator):
    discriminator.trainable = False
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    generated_data = generator([noise, label])
    validity = discriminator([generated_data, label])
    cgan = Model([noise, label], validity)
    cgan.compile(optimizer=Adam(learning_rate, beta_1), loss='binary_crossentropy')
    return cgan

# -----------------------------------
# Veriyi Hazırlama
# -----------------------------------
# TARGET ve TALEP_NO'yu çıkar
X = df_normalized.drop(columns=['TARGET', 'KF_LOAN_NUM']).values
target_values = df_normalized['TARGET'].values
num_samples, feature_dim = X.shape

# -----------------------------------
# CGAN Modellerini Oluşturma
# -----------------------------------
generator = build_generator(latent_dim, feature_dim)
discriminator = build_discriminator(feature_dim)
cgan = build_cgan(generator, discriminator)

# -----------------------------------
# CGAN Eğitim Fonksiyonu
# -----------------------------------
for epoch in range(epochs):
    # Gerçek veriden rastgele bir batch seç
    idx = np.random.randint(0, X.shape[0], batch_size)
    real_data = X[idx]
    real_labels = target_values[idx]

    # Rastgele latent vektörler oluştur ve sahte veri üret
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_labels = np.random.choice([0, 1], batch_size)
    generated_data = generator.predict([noise, fake_labels])

    # Discriminator için gerçek ve sahte veriyi birleştir
    data = np.vstack([real_data, generated_data])
    labels = np.hstack([real_labels, fake_labels])
    validity = np.hstack([np.ones(batch_size), np.zeros(batch_size)])

    # Discriminator'ı eğit
    d_loss = discriminator.train_on_batch([data, labels], validity)

    # Generator'ı eğit
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    random_labels = np.random.choice([0, 1], batch_size)
    g_loss = cgan.train_on_batch([noise, random_labels], np.ones(batch_size))

    # Her 100 epoch'ta bir kayıp değerlerini yazdır
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - D Loss: {d_loss[0]}, D Acc: {d_loss[1]} - G Loss: {g_loss}")

# -----------------------------------
# Sentetik Veri Üretimi
# -----------------------------------
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_labels = np.random.choice([0, 1], num_samples)
synthetic_data = generator.predict([noise, synthetic_labels])

# Sentetik veriyi DataFrame'e dönüştür
synthetic_df_cgan_2 = pd.DataFrame(synthetic_data, columns=df_normalized.drop(columns=['TARGET', 'KF_LOAN_NUM']).columns)
synthetic_df_cgan_2['TARGET'] = synthetic_labels

# Orijinal TALEP_NO dağılımını kontrol et ve ekle
synthetic_talep_no = np.random.choice(df_normalized['KF_LOAN_NUM'].unique(), size=num_samples, replace=True)
synthetic_df_cgan_2['KF_LOAN_NUM'] = synthetic_talep_no

# -----------------------------------
# Sonuçları İnceleme
# -----------------------------------
print("Sentetik TARGET Dağılımı:")
print(synthetic_df_cgan_2['TARGET'].value_counts(normalize=True))

print("\nSentetik TALEP_NO Dağılımı:")
print(synthetic_df_cgan_2['KF_LOAN_NUM'].value_counts(normalize=True))


X_train, y_train, X_valid, y_valid, X_test, y_test, train_data, test_data, valid_data = prepare_data_by_group(
    df=synthetic_df_cgan_2,
    group_column="KF_LOAN_NUM",
    columns=columns,
    target_column="TARGET",
    test_size=0.2,
    val_size=0.2,
    random_state=42
)


logistic_model = LogisticRegression(random_state=42)

# Fonksiyonu çağırma
train_and_evaluate_model(
    model=logistic_model,
    model_name="Logistic Regression",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test
)


import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

# -----------------------------------
# VAE Parametreleri
# -----------------------------------
input_dim = df_normalized.drop(columns=['TARGET', 'KF_LOAN_NUM']).shape[1]  # Özellik sayısı
latent_dim = 32  # Latent space boyutu
epochs = 100  # Eğitim için epoch sayısı
batch_size = 64  # Mini-batch boyutu
learning_rate = 0.001  # Öğrenme oranı

# -----------------------------------
# VAE Modelini Tanımlama
# -----------------------------------

# Sampling fonksiyonu
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(64, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model(inputs, z_mean, name="encoder")

# Decoder
latent_inputs = Input(shape=(latent_dim,))
h_decoded = Dense(64, activation='relu')(latent_inputs)
outputs = Dense(input_dim, activation='sigmoid')(h_decoded)

decoder = Model(latent_inputs, outputs, name="decoder")

# VAE
outputs = decoder(encoder(inputs))
vae = Model(inputs, outputs, name="vae")

# Kayıp fonksiyonu
reconstruction_loss = K.sum(K.binary_crossentropy(inputs, outputs), axis=-1)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(learning_rate))

# -----------------------------------
# Veriyi Hazırlama
# -----------------------------------
X = df_normalized.drop(columns=['TARGET', 'KF_LOAN_NUM']).values
target_values = df_normalized['TARGET'].values
num_samples = X.shape[0]

# -----------------------------------
# VAE'yi Eğitme
# -----------------------------------
vae.fit(X, epochs=epochs, batch_size=batch_size, shuffle=True)

# -----------------------------------
# Sentetik Veri Üretimi
# -----------------------------------
latent_vectors = np.random.normal(size=(num_samples, latent_dim))  # Rastgele latent vektörler
synthetic_data = decoder.predict(latent_vectors)

# Sentetik veriyi DataFrame'e dönüştür
synthetic_df_vae = pd.DataFrame(synthetic_data, columns=df_normalized.drop(columns=['TARGET', 'KF_LOAN_NUM']).columns)

# Orijinal TARGET ve TALEP_NO ekleme
synthetic_targets = np.random.choice(target_values, size=num_samples, replace=True)
synthetic_talep_no = np.random.choice(df_normalized['KF_LOAN_NUM'].unique(), size=num_samples, replace=True)

synthetic_df_vae['TARGET'] = synthetic_targets
synthetic_df_vae['KF_LOAN_NUM'] = synthetic_talep_no

# -----------------------------------
# Sonuçları İnceleme
# -----------------------------------
print("Sentetik TARGET Dağılımı (VAE):")
print(synthetic_df_vae['TARGET'].value_counts(normalize=True))

print("\nSentetik TALEP_NO Dağılımı (VAE):")
print(synthetic_df_vae['KF_LOAN_NUM'].value_counts(normalize=True))


X_train, y_train, X_valid, y_valid, X_test, y_test, train_data, test_data, valid_data = prepare_data_by_group(
    df=synthetic_df_vae,
    group_column="KF_LOAN_NUM",
    columns=columns,
    target_column="TARGET",
    test_size=0.2,
    val_size=0.2,
    random_state=42
)


logistic_model = LogisticRegression(random_state=42)

# Fonksiyonu çağırma
train_and_evaluate_model(
    model=logistic_model,
    model_name="Logistic Regression",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test
)
