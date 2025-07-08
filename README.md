1. **Breaking BERT: Understanding its Vulnerabilities for Named Entity Recognition through Adversarial Attack**  
   1. Chart models for vulnerabilities on NER  
   2. Which type of BERT models are most at risk  
   3. Methods  
      1. Synonym Replacement  
      2. Entity Replacement  
      3. Aims  
         1. Create adversarial examples in which the target entity is no longer recognizable.  
      4. Metrics  
      5. ECA  
         1. Word Perturbations   
         2. Gathering synonyms  
         3. Filtering  
         4. Selection  
   4. Experiment  
      1. Data  
      2. Models  
   5. Results  
      1. ECA  
         1. Entities partially or fully wrong \~87%  
         2. \~75% of the BERT models were fooled by a single change.  
      2. EA  
         1. \~90-100% attract success rate.

2. **Expanding Scope: Adapting English Adversarial Attacks to Chinese**  
   1. Can SOTA attack methods generalize to other languages? Focus on English and Chinese.  
   2. Method  
      1. A goal function  
      2. Set of constraints  
      3. A suite of transformation  
      4. A search algorithm  
   3. Word Synonym Perturbation  
      1. Generation  
         1. HowNet  
         2. MLM  
      2. Transformations  
         1. Homophone   
         2. Morphonym  
         3. Composite  
   4. Results and Evaluation  
      1. Classification and Entailment   
         1. Composite scored the highest  
         2. Morphonym the lowest

3. **Adversarial Demonstration Attacks on Large Language Models**  
   1. Propose a simple and effective in-context learning attack method named advICL.  
   2. DIfferent from text-example based. Introduce a demonstration-specific similarity method that applies constraints on each individual demonstration.  
   3. Related work   
      1. In-Context Learning  
         1. Performance impact  
            1. Demonstration ordering  
            2. Demonstration selection  
            3. Label selection for demonstration  
      2. Adversarial Attacks  
   4. Method  
      1. advICL  
         1. Word level  
         2. Character level  
   5. Experiment  
      1. Models and Dataset  
      2. ICL and Attack Setting  
      3. Effectiveness of advICL  
      4. Evaluation  
         1. ASR  
         2. Clean Accuracy  
      5. Ablation   
         1. Different Template  
         2. Transferability  
   6. Conclusion  
      1. Using advICL shows demonstration attacks on ICL are vulnerable and transferable.

4. **Adversarial Attacks on Tables with Entity Swaps**  
   1. Propose an evasive entity-swap attack for column type annotation tasks. CTA task of a Tabular language model TaLM.  
   2. TaLM has shown SOTA on table interpretation tasks.  
      1. Entity Linking  
      2. Column Type Annotation  
   3. Entity swap \~20-100% of Column Entities  
   4. CTA Adversarial Attack  
      1. Model  
      2. Entity Swap  
      3. Key Entities  
      4. Adversarial Entities  
   5. Evaluation  
      1. Importance Score  
      2. Sampling  
      3. Perturbing Metadata

5. **Gradient-Based Adversarial Attacks against Text Transformers**  
   1. Propose general purpose gradient base attack on a transformer model.  
   2. GBDA  
      1. Gradient-based Distribution Attack  
         1. Instead of a single adversarial example, search for adversarial distributions.  
         2. Enforce perceptibility and fluency.  
      2. Extension to probability vector inputs  
      3. Gumbel-softmax  
      4. Soft Constraints  
   3. Experiment  
      1. Setup  
         1. Tasks  
         2. Models  
         3. Baseline  
         4. Hyperparameters  
      2. White-box  
         1. 1k inputs and 100 examples  
         2. Transfer Attack  
   4. Conclusion  
      1. Highly Effective on a variety of tasks, architectures, and settings.

6. **Why Should Adversarial Perturbations be Imperceptible? Rethink the Research Paradigm in Adversarial NLP**  
   1. Propose “SoadNLP” Security-oriented adversarial NLP.  
      1. Security  
      2. Evaluation  
      3. Explainability  
      4. Augmentation  
   2. Release dataset  
      1. Advbench  
         1. Security Dataset Collection  
         2. Construction  
            1. Motivation  
            2. Tasks  
               1. Misinformation  
               2. Disinformation  
               3. Toxic  
               4. Span  
               5. Sensitive Information Disclosure  
   3. Formalization  
   4. Method  
      1. ROCKET  
   5. Experiment  
      1. Evaluation  
         1. Attack Success Rate  
         2. Average Query Times  
      2. Baseline  
      3. Results  
         1. Some attacks are tough to manual craft.  
         2. Great performance and efficiency compared to baselines.  
      4. Defense Proposals  
         1. Training data augmentation  
         2. Adversarial training  
         3. Preprocessing module  
         4. Robust representation learning  
         5. Certified robustness

7. **Gradient-Based Word Substitutions for Obstinate Adversarial Examples Generation in Language Models**  
   1. Introduce GradObstinant a gradient based approach using word substitution method that automatically generates obstinate adversarial examples.  
   2. Related Work  
      1. Fickle Adversarial Robustness  
      2. Obstinate Adversarial Robustness  
   3. Approach  
      1. Find an obstinate attack on a LLM’s on a sentence input.  
      2. Expand search space.  
      3. Ensuring valid vocabulary.   
   4. Experiment  
      1. Settings  
         1. Benchmarks  
         2. Tasks  
         3. Victim Models  
      2. Choose the target word.  
         1. Perturb most important words in sentences because the least important would be meaningless.  
      3. Threat Model  
         1. Access to model  
         2. Access to data  
      4. State of the Art  
         1. A100 GPU  
   5. Evaluation  
      1. Single Sentence  
         1. Significantly outperforms antonym based strategy. \~97% vs \~50%  
      2. Multi-sentence task  
         1. Significantly outperforms antonym based strategy. \~95% vs \~30%  
      3. Cross-Model  
         1. Successful ASR but loss in accuracy  
      4. Transferability  
         1. Obstinate   
            1. Model  
               1. Independent of model.  
            2. Context  
               1. Might be affected by context.

8. **Sample Attackability in Natural Language Adversarial Attacks**  
   1. Effect to identify robust attack samples to explain a lack of portability attack detection methods.  
   2. Extensive efforts have been put forth to standards for defense either around adversarial training or independent detection systems. Trying to understand empirically how close a random attack sample would be to a decision boundary.  
   3. Attacks  
      1. Inputs to outputs variation  
   4. Experiment  
      1. Setup  
         1. Understand how well a deep-learning based detector performs at detecting an attack on unseen data and unseen models.  
      2. Attacks  
         1. TextFooler  
         2. Bert-based Adversarial Examples  
         3. Improved Genetic Algorithm  
         4. Probability Weighted Word Saliency  
      3. Imperceptability  
         1. Pre-transformation  
         2. Distance  
   5. Results  
      1. Complex  
      2. Portability  
         1. Deep-learning based detection is better than an uncertainty approach.   
         2. Can identify attacks on unseen models but for known attacks but not for unknown attacks.

9. **Bad Characters: Imperceptible NLP Attacks**  
   1. Explore using imperceptible encoding-specific perturbations in a black-box setting.  
   2. Attack Taxonomy  
      1. Invisible Characters  
      2. Homoglyphs  
      3. Reordering  
      4. Deletions  
   3. Methodology  
      1. Optimization problem  
      2. Access to Model outputs  
      3. Integrity attack  
      4. Availability attack  
      5. Restricted Budget  
   4. Evaluation  
      1. Machine Translation  
      2. Toxic Content Detection  
      3. Textual Entailment  
      4. NER  
      5. Sentiment Analysis  
   5. Attack Potential Conclusions  
      1. Imperceptible perturbations derived from manipulating Unicode encodings provide a broad and powerful class of attacks on text-based NLP models.  
   6. Potential Defenses  
      1. OCR  
      2. Clean Data  
      3. Mapping Dictionary

10. **Adversarial Examples Are Not Bugs, They Are Features**  
    1. Present a simple setting to observe misalignment between human-specific notion of robustness and the inherent geometry of data. And claim Adversarial vulnerabilities are a direct result of model sensitivity to well-generalized features.   
    2. Finding Features  
       1. Robust   
       2. Non-Robust  
    3. Training  
       1. Standard  
       2. Robust  
    4. Disentagling   
    5. Transferability  
       1. Can arise from non-robust features  
    6. Conclusion  
       1. Provide support for a phenomenon of adversarial examples as a natural consequence of highly-predictive, non-robust features.

11. **An LLM Assisted Exploitation of AI-Guardian**  
    1. Look into the robustness of AI-Guardian, a defense to adversarial examples. Allows for human-codeless attacks leveraging GPT-4 by prompting it to generate attack methods.  
    2. Uses a public model with and applies secret transformations that render server side. So that if it remains hidden from the adversary the model is protected, but this paper presents effective attacks to reveal the transformation.  
    3. Threat Model  
       1. Distortion Bounds  
       2. Linfinity  
       3. MNIST assuming gradient access to the model  
       4. Can’t access permutation or backdoor trigger functions  
    4. Extraction Attack  
       1. Mask Recovery  
       2. Permutation Recovery  
       3. Pattern Recovery  
    5. Evaluation  
       1. Train One-Hundred models with random mask and pattern  
       2. ASR \~90%

12. **Universal and Transferable Adversarial Attacks on Aligned Language Models**  
    1. Out of the box LLM’s strive to reach alignment to prevent undesirable generation. Propose suffix appended to multiple prompts as well as models.  
    2. Token Elements  
       1. Initial Affirmative Response  
       2. Combined greedy and gradient-based optimization  
       3. Robust Multi-prompt and Multi-model attacks  
    3. Experiment  
       1. Harmful Strings   
       2. Harmful Behaviors  
       3. Metrics  
       4. Baselines  
    4. Results  
       1. ASR \~88% on strings  
       2. ASR \~100% on behavior  
    5. Transferability  
       1. Able to maintain successful ASR on a range on models  
    6. Related Work  
       1. Alignment approaches to LLM’s  
       2. Adversarial Transferability  
       3.  Optimization and Automatic Prompt Tuning

13. **Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To**  
    1. Explores the effectiveness at compromising safety alignments on LLM’s by fine-tuning with only a few adversarial training examples. And also uncovered even benign and commonly used dataset can reduce the ability of guardrails.  
    2. Risk  
       1. Level one  
          1. Fine-tuning with explicitly harmful datasets  
       2. Level Two  
          1. Fine-tuning with implicitly harmful data  
       3. Level Three  
          1. Fine-tuning with benign data  
    3. Fine-Tuning Alignment  
       1. Attackers  
          1. Capability  
          2. Objective  
       2. Setup  
       3. Safety Benchmarks  
    4. Automatic Evaluations  
    5. Metrics  
    6. System Prompts  
    7. Attacks  
       1. Demonstration  
       2. Identity Shifting  
       3. Benign  
    8. Mitigation and Implications  
       1. Techniques  
          1. Pre-training and alignment  
          2. Fine-tuning data moderation  
          3. Safety Auditing  
       2. Law and Policy  
    9. Conclusion  
       1. Safety Alignments from pre-trained models can be circumnavigated intentionally or inadvertently through fine-tuning 

