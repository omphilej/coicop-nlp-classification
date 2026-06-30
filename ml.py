import os
import pandas as pd  # Data manipulation
import numpy as np   # Numerical operations
import fasttext      # FastText embeddings library
import lightgbm as lgb  # LightGBM classifier library
import re            # Regular expressions for text cleaning
import pickle        # Model serialization
from sklearn.model_selection import train_test_split  # Train-test splitting
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder  # Encode target variable
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.metrics import classification_report, f1_score  # Model evaluation
from sklearn.utils.class_weight import compute_sample_weight  # Class weighting
import warnings      # Warning suppression
from sklearn.exceptions import UndefinedMetricWarning  # Specific warning type
from sklearn.model_selection import StratifiedKFold, cross_validate  # Cross-validation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Suppress UndefinedMetricWarning which appears for rare classes with zero predictions
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

print("="*80)
print("SECTION 1: LOAD AND PREPROCESS DATA")
print("="*80)

# Load CSV file containing item descriptions, barcodes, categories, and COICOP codes
# Expected columns: 'Item' (product name), 'Barcode', 'Category', 'COICOP' (target)
df = pd.read_csv('_01SHOPRITE23.csv')
print(f"✓ Loaded data: {len(df)} rows, {df.shape[1]} columns")

# Define a function to clean text for FastText embeddings
def clean_text(text):
    # Convert to lowercase for case-insensitive matching
    text = str(text).lower()
    
    # Remove punctuation: replace anything that's not a-z, 0-9, or space with space
    # This handles special characters, accents, symbols, etc.
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace: replace multiple spaces with single space, then strip ends
    # Prevents issues with "word  word" becoming separate tokens
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply text cleaning function to all product descriptions
# This standardizes input before FastText training
df['item_cleaned'] = df['Item'].apply(clean_text)
print(f"✓ Cleaned {len(df)} item descriptions")

# Convert categorical features to category dtype for efficient storage
# LightGBM can natively handle categorical features without one-hot encoding
cat_features = ['Barcode', 'Category']
for col in cat_features:
    df[col] = df[col].astype('category')
print(f"✓ Converted categorical features: {cat_features}")

# Encode the target variable (8-digit COICOP codes) to numeric labels
# LightGBM requires integer class labels (0, 1, 2, ..., n_classes-1)
target_le = LabelEncoder()  # Creates encoder mapping COICOP → integers
df['target'] = target_le.fit_transform(df['COICOP'].astype(str))  # Fit and transform
num_classes = len(target_le.classes_)  # Count unique classes
print(f"✓ Encoded {num_classes} unique COICOP classes")

print("\n" + "="*80)
print("SECTION 2: TRAIN FASTTEXT UNSUPERVISED MODEL")
print("="*80)

# Write cleaned items to text file (required format for FastText)
# FastText requires one sentence per line for unsupervised training
with open('items_for_ft.txt', 'w', encoding='utf-8') as f:
    for item in df['item_cleaned']:
        # Write each cleaned item description on a new line
        f.write(f"{item}\n")
print(f"✓ Wrote {len(df)} items to items_for_ft.txt")

# Train FastText skipgram model to learn semantic embeddings
# Skipgram learns to predict context words from target word (reverse of CBOW)
# This captures semantic relationships: similar products → similar embeddings
ft_model = fasttext.train_unsupervised(
    'items_for_ft.txt',                  # Input text file (one item description per line)
    model='skipgram',                    # Architecture: skipgram learns context from center word
    wordNgrams=3,                         # Use character n-grams up to 3 chars for subword info
    dim=300,                              # Output embedding dimension (300 = standard size)
    epoch=30,                             # Training epochs (30 = reasonable for 8,823 items)
    lr=0.01,                              # Learning rate (0.01 = conservative, prevents instability)
    verbose=False,                       # Suppress training output (don't print progress)
    minCount=1,                            # Include all words even if they appear only once
    minn=3                                # Minimum character n-gram length (3-char min)
)
print(f"✓ Trained FastText model with 300-dim embeddings")

print("\n" + "="*80)
print("SECTION 3: GENERATE FASTTEXT EMBEDDINGS")
print("="*80)

# Function to get sentence vector from FastText model
def get_vector(text):
    # get_sentence_vector averages word vectors, handles OOV words via char n-grams
    return ft_model.get_sentence_vector(text)

# Generate embeddings for all cleaned items
# This creates a 8823 x 300 array where each row is an item's embedding
embeddings = np.array([get_vector(t) for t in df['item_cleaned']])
print(f"✓ Generated {embeddings.shape[0]} embeddings of size {embeddings.shape[1]}")

# Create DataFrame with embeddings and proper column names
# Each column represents one dimension of the embedding space
emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]  # ['emb_0', 'emb_1', ..., 'emb_299']
df_embeddings = pd.DataFrame(embeddings, columns=emb_cols)  # Convert numpy array to DataFrame

print("\n" + "="*80)
print("SECTION 4: FILTER OUT SINGLE-MEMBER CLASSES (NOISE)")
print("="*80)

# Remove classes with only 1 sample (too rare to learn from or validate)
# These are noise classes that can't be properly trained/evaluated
initial_classes = len(df['target'].unique())  # Count initial unique classes
df = df.groupby('target').filter(lambda x: len(x) > 1)  # Keep groups with >1 sample
final_classes = len(df['target'].unique())  # Count remaining unique classes
print(f"✓ Filtered classes: {initial_classes} → {final_classes} (removed {initial_classes - final_classes} single-member classes)")

# Re-ensure categorical dtype after filtering (pandas sometimes drops this)
# Important for LightGBM to recognize categorical features
for col in cat_features:
    df[col] = df[col].astype('category')

# Re-generate embeddings only for filtered data (maintain alignment with df)
# We need to ensure embedding indices match filtered dataframe indices
embeddings = np.array([get_vector(t) for t in df['item_cleaned']])
df_embeddings = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
print(f"✓ Dataset size after filtering: {len(df)} rows")

print("\n" + "="*80)
print("SECTION 5: PREPARE FEATURE SET (COMBINE CATEGORICAL + EMBEDDINGS)")
print("="*80)

# Combine categorical features (Barcode, Category) with FastText embeddings
# This creates a feature matrix with both metadata and semantic information
X = pd.concat([df[cat_features], df_embeddings], axis=1)  # Horizontal concatenation
y = df['target']  # Target variable (numeric COICOP codes)
print(f"✓ Feature set shape: {X.shape}")
print(f"  - {len(cat_features)} categorical features")
print(f"  - {len(emb_cols)} embedding features")

print("\n" + "="*80)
print("SECTION 6: TRAIN-TEST SPLIT WITH STRATIFICATION")
print("="*80)

# Split data: 80% train, 20% test
# Stratification ensures each class is represented proportionally in both sets
# This is crucial for imbalanced multi-class problems
X_train, X_test, y_train, y_test = train_test_split(
    X,              # Feature matrix
    y,              # Target vector
    test_size=0.2,  # 20% for testing (standard practice)
    random_state=42,  # Fix random seed for reproducibility
    stratify=y      # Stratify by target to maintain class distribution
)
print(f"✓ Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Test set:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

print("\n" + "="*80)
print("SECTION 7: PCA DIMENSIONALITY REDUCTION")
print("="*80)

# Apply PCA to reduce 300 embedding dimensions to 80
# Why: Embeddings are high-dimensional and contain noise
# PCA retains 94.65% of variance while reducing from 300 to 80 dims
# This reduces overfitting on rare classes and speeds up training
pca = PCA(n_components=80)  # Reduce to 80 components
# Fit PCA on training embeddings (separate from categorical features)
X_train_emb = X_train[emb_cols]  # Extract only embedding features
X_test_emb = X_test[emb_cols]    # Extract only embedding features
X_train_emb_pca = pca.fit_transform(X_train_emb)  # Fit on train, transform train
X_test_emb_pca = pca.transform(X_test_emb)  # Transform test using train fit
print(f"✓ PCA: 300 → 80 dimensions")
print(f"✓ Explained variance: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

# Create PCA column names for the reduced features
# These will be column names in the final feature matrix
pca_cols = [f'pca_{i}' for i in range(X_train_emb_pca.shape[1])]  # ['pca_0', ..., 'pca_79']

# Combine categorical features with PCA-reduced embeddings
# Final feature set = 2 categorical + 80 PCA = 82 total features
X_train_pca = pd.concat([
    X_train[cat_features].reset_index(drop=True),  # Categorical features
    pd.DataFrame(X_train_emb_pca, columns=pca_cols)  # PCA-reduced embeddings
], axis=1)

X_test_pca = pd.concat([
    X_test[cat_features].reset_index(drop=True),
    pd.DataFrame(X_test_emb_pca, columns=pca_cols)
], axis=1)

print(f"✓ Train set with PCA: {X_train_pca.shape}")
print(f"✓ Test set with PCA:  {X_test_pca.shape}")

# Reset indices for proper alignment between X and y during training
# Important: LightGBM expects aligned indices
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print("\n" + "="*80)
print("SECTION 8: COMPUTE CLASS WEIGHTS FOR IMBALANCED DATA")
print("="*80)

# Compute balanced sample weights (rare classes get higher weight)
# In imbalanced classification, rare classes are underrepresented
# Class weights compensate by making rare samples more "important" during training
sample_weights_train = compute_sample_weight('balanced', y_train)
# 'balanced' mode: weight = n_samples / (n_classes * class_count)
# Rare classes get high weight, common classes get low weight
print(f"✓ Computed balanced class weights")
print(f"  - Min weight: {sample_weights_train.min():.4f}")  # Most common class weight
print(f"  - Max weight: {sample_weights_train.max():.4f}")  # Rarest class weight
print(f"  - Weight range: {sample_weights_train.max() / sample_weights_train.min():.2f}x")  # Ratio

print("\n" + "="*80)
print("SECTION 9: TRAIN LIGHTGBM CLASSIFIER")
print("="*80)

# Initialize LightGBM multiclass classifier
# LightGBM (Light Gradient Boosting Machine) is a fast, efficient gradient boosting library
# Advantages: handles categorical features natively, fast training, good accuracy
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',           # Multiclass classification (246 classes)
    num_class=len(np.unique(y)),      # Number of output classes
    num_leaves=10,                    # Tree depth complexity (low = simpler trees, less overfit)
    learning_rate=0.01,               # Learning rate (0.01 = conservative, stable)
    n_estimators=10000,                # Max number of boosting rounds (trees to train)
    importance_type='gain',          # Feature importance based on gain (total reduction in loss from splits)
    min_data_in_leaf=2,               # Minimum samples per leaf (low = allows small groups)
    class_weight='balanced',          # Built-in class balancing (alternative to sample weights)
    verbose=-1                        # Suppress training output (-1 = no output)
)

# Train on PCA-reduced features with sample weights
# fit() learns the model from training data
lgb_model.fit(
    X_train_pca,                            # Feature matrix
    y_train,                                # Target vector
    sample_weight=sample_weights_train,     # Class weights (rare classes weighted higher)
    categorical_feature=cat_features,
    eval_set=[(X_test_pca, y_test)],       # Validation set (monitor test loss during training)
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),  # Stop if val loss doesn't improve for 2100 rounds
        lgb.log_evaluation(period=200)            # Log metrics every 200 iterations
    ]
)
print(f"✓ Model trained successfully")

print("\n" + "="*80)
print("SECTION 10: MAKE PREDICTIONS")
print("="*80)

# Get predicted class labels (deterministic: argmax of probabilities)
# Returns one class per sample (0-245 corresponding to 246 COICOP codes)
y_pred = lgb_model.predict(X_test_pca)
print(f"✓ Generated predictions for {len(y_pred)} test samples")

# Get prediction probabilities (needed for confidence analysis)
# Returns shape (n_samples, n_classes) with probabilities summing to 1.0
y_pred_proba = lgb_model.predict_proba(X_test_pca)
print(f"✓ Generated probability scores: shape {y_pred_proba.shape}")

print("\n" + "="*80)
print("SECTION 11: CLASSIFICATION REPORT (ALL CLASSES IN TEST SET)")
print("="*80)

# Get unique classes in test set and their original COICOP codes
# Need to map back from numeric labels to original COICOP code strings
unique_classes_test = np.unique(y_test)  # Classes that appear in test set
class_names = target_le.inverse_transform(unique_classes_test)  # Convert to COICOP codes

# Print detailed classification report
# Shows precision, recall, F1-score for each class and overall metrics
print(classification_report(
    y_test,                  # True labels
    y_pred,                  # Predicted labels
    labels=unique_classes_test,  # Classes to report on
    target_names=class_names.astype(str),  # COICOP codes as labels
    digits=4,                # Show 4 decimal places
    zero_division=0          # Handle classes with 0 support (return 0 instead of error)
))

print("\n" + "="*80)
print("SECTION 12: CROSS-VALIDATION ON FULL DATASET")
print("="*80)

# This model will be trained during cross-validation (separate from lgb_model above)
# Build proper pipeline
preprocessor = ColumnTransformer([
    ('cat_encoded', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_features),
    ('pca_embeddings', PCA(n_components=80), emb_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y)),
        num_leaves=10,
        learning_rate=0.01,
        n_estimators=10000,
        min_data_in_leaf=2,   
        importance_type='gain', 
        class_weight='balanced',
        Verbose=-1
    ))
])

# Now CV is clean - PCA refitted on each fold
cv_results = cross_validate(
    pipeline, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['f1_weighted', 'accuracy'],
    n_jobs=-1, 
)

print(f"✓ 5-Fold Cross-Validation completed")
# Extract and print CV results
print(f"  CV F1 (weighted): {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}")
print(f"  CV Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
# Print individual fold scores
print(f"  Fold scores: {[f'{score:.4f}' for score in cv_results['test_f1_weighted']]}")

print("\n" + "="*80)
print("SECTION 13: FILTERED EVALUATION")
print("="*80)

# Calculate class distribution in test set
# Some classes may have very few test samples, making metrics unreliable
class_counts = pd.Series(y_test).value_counts()  # Count samples per class

# Filter to classes with at least min_support samples
# Avoids reporting metrics for classes with insufficient test data
min_support = 2  # Only evaluate classes with ≥2 test samples
classes_with_support = class_counts[class_counts >= min_support].index.values

# Create boolean mask for filtering
# True where class has sufficient support, False otherwise
mask = pd.Series(y_test).isin(classes_with_support).values
y_test_filtered = y_test[mask]  # Keep only predictions from well-represented classes
y_pred_filtered = y_pred[mask]

# Calculate F1 scores only on well-represented classes
# This gives more realistic performance estimate
f1_macro = f1_score(y_test_filtered, y_pred_filtered, average='macro', zero_division=0)
# macro = unweighted mean of F1 across classes (all classes equally important)
f1_weighted = f1_score(y_test_filtered, y_pred_filtered, average='weighted', zero_division=0)
# weighted = F1 weighted by class support (larger classes more important)

print(f"✓ Classes evaluated: {len(classes_with_support)} (support >= {min_support})")
print(f"✓ Test samples used: {len(y_test_filtered)} / {len(y_test)}")
print(f"  F1 (macro):   {f1_macro:.4f}")
print(f"  F1 (weighted): {f1_weighted:.4f}")

print("\n" + "="*80)
print("SECTION 14: PREDICTION CONFIDENCE ANALYSIS")
print("="*80)

# Extract maximum probability for each prediction
# Confidence = max(probability) = how sure the model is
confidence = y_pred_proba.max(axis=1)  # Max probability across all classes
print(f"✓ Confidence scores computed")
print(f"  Min confidence: {confidence.min():.4f}")  # Least confident prediction
print(f"  Max confidence: {confidence.max():.4f}")  # Most confident prediction
print(f"  Mean confidence: {confidence.mean():.4f}")  # Average confidence
print(f"  Median confidence: {np.median(confidence):.4f}")  # Median confidence

# Create binary mask: confident vs uncertain predictions
# Predictions above threshold are "confident" and auto-classified
# Predictions below threshold are "uncertain" and sent for manual review
# Find optimal threshold based on business needs
thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

print(f"{'Threshold':<12} {'Auto %':<10} {'Accuracy':<12} {'Manual %':<10}")
print("-" * 50)

for thresh in thresholds:
    auto_mask = confidence > thresh
    if auto_mask.sum() > 0:
        auto_acc = (y_pred[auto_mask] == y_test[auto_mask]).mean()
        auto_pct = auto_mask.mean() * 100
        manual_pct = 100 - auto_pct
        print(f"{thresh:<12.2f} {auto_pct:<10.1f} {auto_acc*100:<12.2f} {manual_pct:<10.1f}")
        
threshold = 0.80  # Example threshold: 80% confidence or higher is auto-classified
certain_mask = confidence > threshold  # Boolean: True if confident
uncertain_mask = ~certain_mask  # Boolean: True if uncertain (inverse of certain)

# Count and percentage of confident predictions
n_confident = certain_mask.sum()  # Number of confident predictions
n_uncertain = uncertain_mask.sum()  # Number of uncertain predictions
pct_confident = (n_confident / len(y_test)) * 100  # Percentage confident
pct_uncertain = (n_uncertain / len(y_test)) * 100  # Percentage uncertain

print(f"\n✓ Confidence threshold: {threshold}")
print(f"  Confident predictions (>{threshold}):   {n_confident} / {len(y_test)} ({pct_confident:.1f}%)")
print(f"  Uncertain predictions (≤{threshold}):  {n_uncertain} / {len(y_test)} ({pct_uncertain:.1f}%)")

# Calculate F1 score for only confident predictions
# Shows performance improvement when filtering by confidence
if n_confident > 0.70 * len(y_test):  # Only if >70% are confident (sample is large enough)
    f1_certain = f1_score(
        y_test[certain_mask],      # True labels of confident predictions
        y_pred[certain_mask],      # Predicted labels of confident predictions
        average='weighted',        # Weighted by class support
        zero_division=0
    )
    print(f"  F1 (weighted, confident only): {f1_certain:.4f}")
    # Compare to overall F1
    f1_all = f1_score(y_test, y_pred, average='weighted')
    improvement = (f1_certain - f1_all) * 100
    print(f"  → Improvement: +{improvement:.2f}%")

print("\n" + "="*80)
print("SECTION 15: ANALYZE FOR MANUAL REVIEW")
print("="*80)

# Get indices and confidence of uncertain predictions
# These are candidates for manual review
uncertain_indices = np.where(uncertain_mask)[0]  # Index positions of uncertain predictions
uncertain_confidence = confidence[uncertain_mask]  # Confidence scores of uncertain predictions

# Sort by confidence (lowest first)
# Show operators which predictions to prioritize (lowest confidence = most uncertain)
sorted_order = np.argsort(uncertain_confidence)  # Indices that would sort the array
sorted_indices = uncertain_indices[sorted_order]  # Reorder uncertain indices by confidence
sorted_confidence = uncertain_confidence[sorted_order]  # Reorder confidences from low to high

print(f"✓ Total uncertain predictions: {len(uncertain_indices)}")
print(f"\nTop 10 LEAST CONFIDENT predictions (candidates for manual review):")
print(f"{'Rank':<6} {'Index':<8} {'Confidence':<15} {'Prediction':<15} {'Actual':<15}")
print("-" * 60)

# Show top 10 least confident predictions
for rank, (idx, conf) in enumerate(zip(sorted_indices[:10], sorted_confidence[:10]), 1):
    # Convert numeric class to COICOP code
    pred_class = target_le.inverse_transform([y_pred[idx]])[0]
    actual_class = target_le.inverse_transform([y_test[idx]])[0]
    print(f"{rank:<6} {idx:<8} {conf:<15.4f} {pred_class:<15} {actual_class:<15}")

print("\n" + "="*80)
print("SECTION 16: FEATURE ANALYSIS")
print("="*80)

# Get feature importances from trained model
# Importance = total contribution of feature to classification across all trees
feature_importance = lgb_model.feature_importances_  # Array of importance scores
feature_names = list(cat_features) + pca_cols  # All feature names

# Get top 20 features
# Show which features the model relies on most
top_20_idx = np.argsort(feature_importance)[-20:]  # Indices of top 20 features
top_features = [(feature_names[i], feature_importance[i]) for i in top_20_idx[::-1]]  # Reverse to descending order

print(f"\nTop 20 Most Important Features:")
print(f"{'Rank':<6} {'Feature':<15} {'Importance':<15}")
print("-" * 40)
for rank, (fname, fimportance) in enumerate(top_features, 1):
    print(f"{rank:<6} {fname:<15} {fimportance:>14.2f}")

# Calculate total importance per feature type
# Compare categorical vs. embedding features
cat_importance_total = feature_importance[:len(cat_features)].sum()  # Sum of categorical importances
pca_importance_total = feature_importance[len(cat_features):].sum()  # Sum of PCA importances
total_importance = cat_importance_total + pca_importance_total  # Total of all importances

# Normalize to percentages
cat_importance_pct = (cat_importance_total / total_importance) * 100  # % of total importance
pca_importance_pct = (pca_importance_total / total_importance) * 100

print(f"\n✓ Feature Type Breakdown:")
print(f"  Categorical importance: {cat_importance_pct:.2f}%")
print(f"  PCA embeddings importance: {pca_importance_pct:.2f}%")

print("\n" + "="*80)
print("SECTION 18: PREDICTIONS WITH CONFIDENCE FILTERING")
print("="*80)

# Create a list to store production predictions
# Each item will include: prediction, confidence, status, actual label
production_predictions = []

print(f"\nGenerating production predictions...")
# Iterate through all test samples
for i in range(len(X_test_pca)):
    # Get confidence and predicted class for this sample
    confidence_score = confidence[i]
    predicted_class = y_pred[i]
    
    # Decision logic: confidence threshold determines auto vs. manual
    if confidence_score > threshold:
        # HIGH CONFIDENCE: Auto-classify
        # Convert numeric class back to COICOP code
        predicted_coicop = target_le.inverse_transform([predicted_class])[0]
        status = "AUTO-CLASSIFY"
    else:
        # LOW CONFIDENCE: Flag for manual review
        predicted_coicop = "MANUAL_REVIEW"
        status = "UNCERTAIN"
    
    # Store prediction record
    production_predictions.append({
        'index': i,                    # Sample index
        'predicted_coicop': predicted_coicop,  # Predicted COICOP code or "MANUAL_REVIEW"
        'confidence': confidence_score,  # Confidence score (0-1)
        'status': status,              # "AUTO-CLASSIFY" or "UNCERTAIN"
        'actual_coicop': target_le.inverse_transform([y_test[i]])[0]  # True COICOP code
    })

# Convert to DataFrame for easy analysis
prod_df = pd.DataFrame(production_predictions)

# Summary statistics
auto_count = (prod_df['status'] == 'AUTO-CLASSIFY').sum()  # Count auto-classified
manual_count = (prod_df['status'] == 'UNCERTAIN').sum()    # Count for manual review

print(f"\n✓ Production Prediction Summary:")
print(f"  Auto-classify: {auto_count} / {len(prod_df)} ({auto_count/len(prod_df)*100:.1f}%)")
print(f"  Manual review: {manual_count} / {len(prod_df)} ({manual_count/len(prod_df)*100:.1f}%)")

# Accuracy for auto-classified items
auto_mask = prod_df['status'] == 'AUTO-CLASSIFY'  # Boolean mask for auto-classified
auto_correct = (prod_df[auto_mask]['predicted_coicop'] == prod_df[auto_mask]['actual_coicop']).sum()  # Count correct
auto_accuracy = auto_correct / auto_mask.sum() if auto_mask.sum() > 0 else 0  # Calculate accuracy
print(f"  Auto-classify accuracy: {auto_accuracy*100:.2f}%")

# Save production predictions to CSV
# Can be used for deployment, analysis, or as training data for retraining
prod_df.to_csv('production_predictions.csv', index=False)
# After human reviewers correct predictions
def retrain_with_corrections():
    # ────────────────────────────────────────────────────────
    # STEP 1: Check if corrections file exists
    # ────────────────────────────────────────────────────────
    if not os.path.exists('manual_corrections.csv'):
        print("No manual_corrections.csv found - skipping active learning")
        print("  → Run this function after collecting human reviews")
        return None
    
    # ────────────────────────────────────────────────────────
    # STEP 2: Load and combine data
    # ────────────────────────────────────────────────────────
    original_df = pd.read_csv('_01SHOPRITE23.csv')
    corrections_df = pd.read_csv('manual_corrections.csv')
    print(f"✓ Loaded {len(original_df)} original + {len(corrections_df)} corrections")
    
    augmented_df = pd.concat([original_df, corrections_df], ignore_index=True)
    
    # ────────────────────────────────────────────────────────
    # STEP 3: Re-run preprocessing on augmented data
    # ────────────────────────────────────────────────────────
    augmented_df['item_cleaned'] = augmented_df['Item'].apply(clean_text)
    
    for col in cat_features:
        augmented_df[col] = augmented_df[col].astype('category')
    
    # Re-encode targets (in case corrections introduce new classes)
    new_target_le = LabelEncoder()
    augmented_df['target'] = new_target_le.fit_transform(augmented_df['COICOP'].astype(str))
    
    # Filter single-member classes
    augmented_df = augmented_df.groupby('target').filter(lambda x: len(x) > 1)
    
    # ────────────────────────────────────────────────────────
    # STEP 4: Generate embeddings for augmented data
    # ────────────────────────────────────────────────────────
    new_embeddings = np.array([get_vector(t) for t in augmented_df['item_cleaned']])
    new_emb_df = pd.DataFrame(new_embeddings, columns=emb_cols, index=augmented_df.index)
    
    # ────────────────────────────────────────────────────────
    # STEP 5: Apply PCA (refit on augmented data)
    # ────────────────────────────────────────────────────────
    new_pca = PCA(n_components=80)
    new_emb_pca = new_pca.fit_transform(new_emb_df)
    new_pca_cols = [f'pca_{i}' for i in range(80)]
    
    # Combine categorical + PCA features
    X_augmented = pd.concat([
        augmented_df[cat_features].reset_index(drop=True),
        pd.DataFrame(new_emb_pca, columns=new_pca_cols)
    ], axis=1)
    y_augmented = augmented_df['target'].reset_index(drop=True)
    
    # ────────────────────────────────────────────────────────
    # STEP 6: Train-test split on augmented data
    # ────────────────────────────────────────────────────────
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_augmented, y_augmented,
        test_size=0.2, random_state=42, stratify=y_augmented
    )
    
    # ────────────────────────────────────────────────────────
    # STEP 7: Compute weighted samples (corrections weighted 3x)
    # ────────────────────────────────────────────────────────
    # NOTE: Applying weights to TRAINING set only, after split
    # Identify which rows in X_train_new came from corrections
    # (Simplified: use class_weight='balanced' instead for cleaner code)
    sample_weights_new = compute_sample_weight('balanced', y_train_new)
    
    # ────────────────────────────────────────────────────────
    # STEP 8: Create and train NEW model
    # ────────────────────────────────────────────────────────
    new_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y_augmented)),  # Dynamic class count
        num_leaves=10,
        learning_rate=0.01,
        n_estimators=10000,
        min_data_in_leaf=2,
        importance_type='gain',
        class_weight='balanced',
        verbose=-1
    )
    
    new_model.fit(
        X_train_new,
        y_train_new,
        sample_weight=sample_weights_new,
        eval_set=[(X_test_new, y_test_new)],
        categorical_feature=cat_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # ────────────────────────────────────────────────────────
    # STEP 9: Evaluate the new model
    # ────────────────────────────────────────────────────────
    y_pred_new = new_model.predict(X_test_new)
    new_f1 = f1_score(y_test_new, y_pred_new, average='weighted', zero_division=0)
    print(f"\n✓ Retrained model F1 (weighted): {new_f1:.4f}")
    
    # ────────────────────────────────────────────────────────
    # STEP 10: Save the new model + artifacts
    # ────────────────────────────────────────────────────────
    with open('lightgbm_model_v2.pkl', 'wb') as f:
        pickle.dump(new_model, f)
    with open('pca_transformer_v2.pkl', 'wb') as f:
        pickle.dump(new_pca, f)
    with open('label_encoder_v2.pkl', 'wb') as f:
        pickle.dump(new_target_le, f)
    
    print(f"✓ Saved updated model: lightgbm_model_v2.pkl")
    
    return new_model


# ────────────────────────────────────────────────────────
# Call the function (will skip if no corrections file exists)
# ────────────────────────────────────────────────────────
updated_model = retrain_with_corrections() # Now properly defined

print("\n" + "="*80)
print("SECTION 19: SAVE ALL MODELS")
print("="*80)

# Save FastText model
# Required for: generating embeddings for new items
ft_model.save_model('fasttext_model.bin')

# Save LightGBM model
# Required for: making predictions from feature vectors
with open('lightgbm_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)

# Save PCA transformer
# Required for: reducing new embeddings from 300 to 80 dimensions
with open('pca_transformer.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Save label encoder
# Required for: converting numeric class predictions back to COICOP codes
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(target_le, f)

# Save feature names (needed for inference)
# Required for: ensuring new data has correct feature order and names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump({
        'cat_features': cat_features,  # Categorical feature names
        'pca_cols': pca_cols,          # PCA component names
        'emb_cols': emb_cols           # Embedding feature names
    }, f)
    
print(f" models saved")
