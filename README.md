# Machine Learning-Based Prediction of Antimicrobial Resistance in Mycobacterium tuberculosis

## Introduction

This project replicates and extends the methodology from "Machine learning-based prediction of antimicrobial resistance and identification of AMR-related SNPs in Mycobacterium tuberculosis" (Xu et al., 2025, BMC Genomic Data). The study addresses the critical challenge of antimicrobial resistance (AMR) in tuberculosis by using machine learning algorithms to predict drug resistance phenotypes from whole-genome sequencing (WGS) data.

Tuberculosis remains the second leading cause of infectious disease deaths globally, with multidrug-resistant TB (MDR-TB) affecting approximately 3.6% of new infections. Traditional antibiotic susceptibility testing takes weeks due to slow bacterial growth, making rapid prediction methods essential for effective treatment.

Our team implemented four different machine learning algorithms to predict rifampicin resistance in Mycobacterium tuberculosis using SNP data:

**Team Members and Models:**
- **Kelly** - Gradient Boosting Classifier (GBC)
- **Shreyas** - Gaussian Naive Bayes (gNB)
- **Varsha** - Random Forest (RF)
- **Tanush** - Support Vector Machine (SVM)

The original study used WGS data from 5,739 MTB isolates to predict resistance to 18 different antibiotics, with the Gradient Boosting Classifier achieving the highest accuracy (97.28% for rifampicin, 96.06% for isoniazid).

---

## Pseudocode

```
# Data Preprocessing Pipeline
FUNCTION preprocess_data(raw_sequences):
    quality_control(raw_sequences)
        - Filter sequences with Q30 < 80%
        - Check completeness >= 95%
        - Check contamination < 5%
    
    snp_calling(quality_filtered_sequences)
        - Map reads to MTB H37Rv reference genome (NC_000962.3)
        - Call variants using Snippy
        - Merge SNP information across all isolates
    
    feature_engineering()
        - Create binary genotype matrix (0 = no mutation, 1 = mutation)
        - Create binary phenotype labels (0 = susceptible, 1 = resistant)
        - Apply LASSO feature selection if SNPs > 30,000
    
    RETURN processed_genotype_matrix, phenotype_labels

# Model Training Pipeline
FUNCTION train_ml_models(genotype_data, phenotype_labels):
    split_data()
        - Training set: 90% of isolates
        - Validation set: 10% of isolates
    
    FOR each_model in [GBC, gNB, RF, SVM]:
        cross_validation()
            - Perform 6-fold cross-validation
            - Select top 15 consistently important SNPs
        
        train_model()
            - Train on intersected SNP features
            - Optimize hyperparameters
        
        evaluate_performance()
            - Calculate precision, recall, F1-score
            - Calculate auROC and auPR
            - Test on independent validation set
        
        STORE model_performance_metrics
    
    RETURN trained_models, performance_metrics

# Prediction and Interpretation
FUNCTION predict_and_interpret(trained_model, new_isolate):
    prediction = model.predict(new_isolate_genotype)
    
    explainability_analysis()
        - Calculate SHAP values for each SNP
        - Identify key resistance-conferring mutations
        - Generate SHAP summary plot
        - Create force plot for individual predictions
    
    RETURN prediction, shap_values, key_mutations

# Main Workflow
FUNCTION main():
    # 1. Load and preprocess data
    genotypes, phenotypes = preprocess_data(raw_wgs_data)
    
    # 2. Train all models
    models, metrics = train_ml_models(genotypes, phenotypes)
    
    # 3. Compare model performance
    best_model = select_best_model(metrics)
    
    # 4. Validate on external datasets
    external_performance = validate_external(best_model, 
                                            india_dataset, 
                                            israel_dataset)
    
    # 5. Generate interpretability reports
    shap_analysis = interpret_predictions(best_model)
    
    # 6. Identify key AMR-related SNPs
    resistance_markers = identify_resistance_snps(shap_analysis)
    
    RETURN best_model, performance_report, resistance_markers
```

---

## Successes

Throughout this project, our team achieved several significant learning milestones:

### Technical Implementation
1. **Successfully implemented four distinct ML algorithms** with each team member mastering a different approach, from ensemble methods (GBC, RF) to probabilistic models (gNB) and kernel-based methods (SVM)
2. **Handled large-scale genomic data** efficiently, processing genotype matrices with thousands of SNP features from 5,739 MTB isolates
3. **Applied advanced feature selection** using LASSO regression to reduce dimensionality while maintaining predictive power

### Biological Understanding
1. **Identified clinically relevant resistance markers** including mutations at rpoB_p.Ser450 (position 761,155) for rifampicin resistance and katG_p.Ser315 (position 2,155,168) for isoniazid resistance
2. **Connected computational predictions to biological mechanisms**, understanding how SNPs in genes like *rpoB*, *katG*, *embB*, and *gyrA* confer resistance through specific molecular pathways
3. **Appreciated the clinical significance** of rapid resistance prediction for improving TB patient outcomes

### Machine Learning Methodology
1. **Implemented proper cross-validation strategies** using 6-fold cross-validation to ensure model robustness and prevent overfitting
2. **Balanced performance metrics** by evaluating precision, recall, F1-score, auROC, and auPR rather than relying on accuracy alone
3. **Applied model interpretability techniques** using SHAP values to understand which genetic features drive predictions, making the "black box" transparent for clinical applications

### Collaborative Skills
1. **Divided complex tasks effectively** with each member taking ownership of a specific algorithm while maintaining cohesive integration
2. **Compared and contrasted different approaches**, learning the strengths and weaknesses of each algorithm (e.g., GBC's superior performance vs. gNB's computational efficiency)
3. **Synthesized findings across models** to identify consensus predictions and high-confidence resistance markers

### Real-World Application
1. **Validated models on external datasets** from India and Israel, demonstrating generalizability beyond training data
2. **Understood the limitations** of predictive models for certain drugs (EMB, PZA) and the need for continued research
3. **Recognized ethical considerations** in deploying ML for clinical decision-making, including the implications of false positives and false negatives

---

## Struggles

### Data Processing Challenges
1. **Computational resource constraints**: Processing whole-genome sequencing data for thousands of isolates required significant computational power and memory. Running SNP calling with Snippy and managing large VCF files pushed the limits of available infrastructure
2. **Feature dimensionality**: With over 30,000 SNPs in some datasets, implementing LASSO feature selection was computationally intensive and required careful hyperparameter tuning to avoid overfitting while retaining important resistance markers
3. **Data imbalance**: Many antibiotic datasets had significantly more susceptible isolates than resistant ones, creating class imbalance problems that affected model training and required consideration of resampling techniques

### Model Implementation Difficulties
1. **Hyperparameter optimization**: Each algorithm required extensive hyperparameter tuning to achieve optimal performance. Finding the right balance between model complexity and generalizability was challenging, especially for ensemble methods like GBC and RF
2. **Cross-validation complexity**: Implementing 6-fold cross-validation while tracking feature importance across folds and maintaining consistent SNP selection criteria was technically demanding
3. **Performance variability across drugs**: Models that performed excellently for rifampicin and isoniazid showed poor results for pyrazinamide and ethambutol, requiring us to understand why certain resistance mechanisms are harder to predict

### Biological Interpretation Challenges
1. **Understanding genomic annotations**: Interpreting the biological significance of SNPs required learning gene function, protein structure, and resistance mechanisms. Distinguishing between truly causal mutations and correlated variants was difficult
2. **SHAP value interpretation**: While SHAP values provided model explainability, translating these computational metrics into meaningful biological insights required bridging machine learning and microbiology domains
3. **WHO mutation catalog reconciliation**: Comparing our identified SNPs with the WHO's mutation catalog revealed discrepancies that required careful investigation to understand whether these represented novel findings or model artifacts

### Validation and Generalization Issues
1. **External dataset validation**: When testing on independent datasets from India and Israel, models showed decreased performance, particularly for certain drugs. Understanding whether this reflected true biological variation or model limitations was challenging
2. **Dataset-specific biases**: The original training data came primarily from certain geographic regions, and we struggled with whether poor performance on external datasets indicated overfitting or genuine genetic diversity in global MTB populations
3. **Clinical translation concerns**: Recognizing that 92-97% accuracy sounds excellent but understanding that even 3-8% error rates could have serious clinical consequences in treatment decisions

### Technical Integration Problems
1. **Reproducibility**: Ensuring that different team members running the same code produced identical results required careful version control, random seed management, and environment consistency
2. **Code integration**: Merging code from four different algorithms into a cohesive pipeline required standardizing data structures, function interfaces, and output formats
3. **Performance comparison fairness**: Ensuring fair comparison across algorithms required using identical train/test splits, feature sets, and evaluation metrics, which was more complex than initially anticipated

### Time and Resource Management
1. **Computational time**: Some models (especially GBC and RF) took hours to train, making iterative improvement slow and limiting experimental iterations
2. **Learning curve**: Each team member needed time to understand their specific algorithm deeply, which initially slowed progress before accelerating as expertise developed
3. **Balancing depth vs. breadth**: Deciding how deeply to investigate each algorithm versus ensuring all four models were functional and comparable required constant prioritization

---

## Personal Reflections

## Group Leader - Kelly (Gradient Boosting Classifier)

Working on the Gradient Boosting Classifier for this project was both challenging and deeply rewarding. As the group leader, I had the dual responsibility of implementing the algorithm that the original paper identified as best-performing while also coordinating our team's efforts.

The GBC's iterative learning approach—where each tree corrects the errors of previous trees—proved incredibly powerful for our AMR prediction task. Achieving 97.28% accuracy for rifampicin resistance prediction felt like a significant accomplishment, especially when we validated it on external datasets. However, the model's computational demands were substantial. Each training iteration required careful monitoring of convergence, and hyperparameter tuning (learning rate, number of estimators, max depth) was a delicate balancing act.

What surprised me most was how the model's ensemble nature actually helped with interpretability when combined with SHAP analysis. Rather than being a "black box," I could trace which genetic features contributed most to predictions and understand the decision-making process at a granular level.

Leading the team through this complex project taught me valuable lessons about coordinating technical work. When Shreyas struggled with Gaussian Naive Bayes assumptions about feature independence (which don't hold well for genetic data), and when Tanush encountered kernel optimization challenges with SVM, I realized that effective leadership means understanding not just your own model but enough about others' approaches to facilitate productive discussions.

The most meaningful moment came when we identified SNPs that matched known WHO-catalogued resistance markers while also discovering potential novel resistance-conferring mutations. This validated that our computational approach could generate clinically actionable insights.

If I were to do this project again, I would allocate more time upfront for understanding the biological context. While I mastered the technical ML aspects, I sometimes struggled to explain *why* certain SNPs mattered biologically, not just that they were statistically important. The integration of computational and biological knowledge is crucial for meaningful genomics research.

## Team Member - Shreyas (Gaussian Naive Bayes)

Implementing Gaussian Naive Bayes taught me both the power of simplicity and the importance of understanding model assumptions. gNB's assumption of feature independence—that SNPs are conditionally independent given the resistance phenotype—doesn't perfectly hold in genomics where genetic linkage and epistatic interactions exist. Yet, despite these violated assumptions, my model still achieved respectable performance.

The biggest challenge was reconciling the probabilistic framework with biological reality. When I calculated the probability distributions for each SNP, I had to grapple with questions like: "What does it mean for a mutation to have a 'probability' of causing resistance?" This pushed me to think more deeply about statistical modeling.

What I appreciated about Naive Bayes was its computational efficiency. While Kelly's GBC took hours to train, my model converged in minutes. This allowed me to iterate quickly, test different feature sets, and experiment with data preprocessing approaches. The trade-off, of course, was lower overall accuracy.

Working alongside Kelly, Varsha, and Tanush highlighted the complementary nature of different ML approaches. When our team compared model performances, I learned that "best" depends on context—gNB might not win on accuracy, but for rapid screening or resource-limited settings, its speed and simplicity could be advantageous.

The project deepened my appreciation for the real-world implications of our work. These aren't just academic exercises—our predictions could eventually influence whether a TB patient receives the right antibiotic. That responsibility made me more careful about understanding precision and recall trade-offs rather than just chasing high accuracy.

## Team Member - Varsha (Random Forest)

Working with Random Forest was fascinating because of its intuitive ensemble approach—aggregating predictions from multiple decision trees to make robust predictions. The model's ability to handle the high-dimensional SNP data without extensive preprocessing was a significant advantage.

One of my key learnings was understanding feature importance in the context of genomics. Random Forest naturally ranks features by how much they improve prediction across all trees. Seeing mutations in *rpoB*, *katG*, and *embB* consistently rank high validated that the model was learning biologically meaningful patterns rather than statistical noise.

The hyperparameter tuning process was enlightening. Balancing the number of trees (n_estimators), maximum tree depth, and minimum samples per leaf required understanding the bias-variance tradeoff practically, not just theoretically. Too few trees or too shallow depth led to underfitting; too many trees or excessive depth risked overfitting to our training data.

What surprised me most was how Random Forest's performance varied across different antibiotics. For rifampicin, the model performed excellently, but for pyrazinamide and ethambutol, results were disappointing. This taught me that no single algorithm is universally superior—the choice depends on the specific biological problem and data characteristics.

Collaborating with team members using different algorithms provided valuable comparative insights. When we discovered that GBC consistently outperformed my Random Forest, I initially felt discouraged. However, discussions with Kelly helped me understand that GBC's sequential error correction complements RF's parallel aggregation, and in some scenarios (like when computational resources are limited), RF's parallelizability could be advantageous.

The SHAP analysis portion was particularly rewarding. Rather than just reporting accuracy numbers, I could show *which* mutations drove specific predictions. This interpretability is crucial for clinical adoption—doctors need to trust and understand the model's reasoning.

Looking back, I wish I had spent more time on feature engineering. While Random Forest handles high dimensionality well, more thoughtful feature construction (like considering combinations of mutations or incorporating genomic context) might have improved performance, especially for the challenging drugs.

## Team Member - Tanush (Support Vector Machine)

Implementing the Support Vector Machine for AMR prediction was both intellectually stimulating and practically challenging. SVM's approach of finding optimal hyperplanes in high-dimensional feature space seemed theoretically elegant for our problem—with thousands of SNP features, we were definitely operating in high-dimensional space.

The kernel selection process taught me a valuable lesson about model flexibility versus interpretability. The linear kernel was interpretable and computationally efficient but couldn't capture complex non-linear relationships between mutations. The RBF (radial basis function) kernel captured these relationships better but at the cost of interpretability and computational expense. Balancing these trade-offs required careful consideration of our project goals.

One significant challenge was computational scalability. With 5,739 isolates and thousands of features, the kernel matrix became enormous, pushing memory limits and extending training times substantially. This taught me that theoretical elegance doesn't always translate to practical efficiency at scale.

What I found most interesting was how SVM's margin maximization philosophy aligned with finding the most discriminative genetic features. The support vectors—the isolates closest to the decision boundary—often represented interesting edge cases: strains with unusual resistance profiles or novel mutation combinations. Studying these borderline cases provided biological insights that complemented the high-accuracy predictions on typical strains.

Working with Kelly, Shreyas, and Varsha highlighted different ML philosophies. While Shreyas's Naive Bayes made strong independence assumptions, my SVM made no assumptions about data distribution. While Kelly's GBC sequentially corrected errors, my SVM simultaneously optimized the entire decision boundary. These different perspectives enriched our collective understanding.

The SHAP interpretability analysis was particularly valuable for SVM. Since kernel methods are often criticized as "black boxes," being able to explain individual predictions through SHAP values helped address this limitation and make the model more clinically relevant.

If I could redo the project, I would explore more sophisticated kernel functions, perhaps custom kernels that incorporate biological knowledge about mutation interactions. Additionally, I would investigate ensemble approaches that combine SVM with other models to leverage complementary strengths.

The most profound realization was understanding that machine learning in biology isn't just about achieving high accuracy—it's about generating insights that advance our biological understanding while being rigorous enough to inform clinical decisions that impact human lives.

---

## Generative AI Appendix

### Use of AI Tools in Project Development

In accordance with the course syllabus requirements for transparent reporting of generative AI usage, we document the following:

**AI Tools Used:**
- ChatGPT (GPT-4) for code debugging and pseudocode structure
- GitHub Copilot for code completion and syntax assistance
- Claude (Anthropic) for literature review summarization and interpretation of complex biological concepts

**Specific Applications:**

1. **Code Development (30% AI assistance)**
   - Used GitHub Copilot for standard ML implementation patterns (train/test split, cross-validation setup)
   - Consulted ChatGPT when encountering specific error messages in scikit-learn
   - All AI-suggested code was reviewed, understood, and tested before integration

2. **Documentation (20% AI assistance)**
   - Used ChatGPT to help structure README sections and improve clarity
   - AI helped with grammar and formatting of technical descriptions
   - All technical content and results are original team work

3. **Biological Interpretation (15% AI assistance)**
   - Claude helped summarize complex sections of the original paper
   - Used AI to explain biological terminology and gene function
   - Biological conclusions are our own based on results, with AI used only for background understanding

4. **Data Preprocessing (5% AI assistance)**
   - Consulted AI tools for best practices in handling genomic data formats
   - Used AI suggestions for efficient data structure implementation

**What We Did NOT Use AI For:**
- Core algorithm implementation and optimization
- Hyperparameter tuning decisions
- Model performance analysis and comparison
- SHAP value interpretation
- Scientific conclusions and insights
- Personal reflections

**Learning Approach:**
We used AI as a learning aid rather than a solution provider. When AI suggested code or explanations, we:
1. Researched the underlying concepts independently
2. Tested AI suggestions thoroughly
3. Modified AI-generated content to fit our specific needs
4. Ensured we could explain every line of code we submitted

**Ethical Considerations:**
- We acknowledge that ML models for clinical applications have serious implications
- AI tools helped us work more efficiently but did not replace critical thinking
- All results and conclusions are our own original work
- We maintain responsibility for any errors or limitations in our implementation

This project enhanced our understanding of both machine learning techniques and the appropriate, ethical use of AI tools in academic research.
