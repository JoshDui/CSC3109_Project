# CSC3109 Machine Learning Group Project Requirements

**Course:** CSC3109 Machine Learning  
**Academic Year:** AY2025/2026 Trimester 3  
**Project Type:** Group project  
**Final Submission Deadline:** Sunday, 26 July 2026, 23:59 local time  
**Submission Platform:** Designated xSITE Dropbox  
**Submission Rule:** One submission per team

---

## 1. Project Overview

This project requires each team to design, train, evaluate, and deploy a deep learning image-classification system for aerial scene recognition.

The project focuses on translating deep learning concepts into practice by solving a real-world computer vision problem using a curated aerial-imagery dataset. Teams must demonstrate competence across the full machine learning lifecycle, including data exploration, preprocessing, model design, training, evaluation, benchmarking, and deployment.

The final solution must classify aerial images into one of four assigned visually confusable scene categories and must be supported by a professional technical report, reproducible implementation, containerised inference deployment, and a short video demonstration.

---

## 2. Core Objectives

The project objectives are to:

1. Apply deep learning concepts to a real-world computer vision task.
2. Build an aerial-image classification system using modern deep learning frameworks.
3. Conduct meaningful exploratory data analysis on the assigned dataset partition.
4. Prepare, preprocess, and augment image data appropriately.
5. Investigate multiple deep learning approaches.
6. Train, tune, compare, and evaluate image-classification models.
7. Select the best-performing model based on documented evidence.
8. Deploy the selected model using containerisation.
9. Produce a clear technical report documenting methods, results, analysis, limitations, and future work.
10. Demonstrate the model and key findings in a short video.

---

## 3. Key Concepts

The project is expected to involve the following concepts:

- Deep learning
- Convolutional neural networks
- Transfer learning
- Real-world computer vision
- Aerial and satellite imagery analysis
- Data preprocessing
- Data augmentation
- Model benchmarking
- Hyperparameter tuning
- Performance evaluation
- Containerised deployment
- Technical reporting

---

## 4. Problem Context

Deep learning has become important across domains such as remote sensing, urban planning, environmental monitoring, and disaster response.

Aerial and satellite imagery analysis enables intelligent systems to automatically classify land-use types, identify infrastructure patterns, and monitor large-scale geographic changes.

In this project, each team acts as a deep learning team responsible for building a high-performing aerial scene recognition system. The system must be developed with a clear understanding of the data, the classification challenge, and the strengths and weaknesses of the modelling approaches used.

---

## 5. Dataset and Classification Task

### 5.1 Dataset Source

Each team will receive a dedicated dataset partition drawn from a 38-category aerial-imagery benchmark.

### 5.2 Assigned Categories

Each team partition contains exactly **4 visually confusable categories**.

Examples of visually confusable categories may include:

- Different road types
- Different residential area types
- Similar vegetation or land-use categories

The exact assigned categories are team-specific.

### 5.3 Dataset Size

For each of the 4 assigned categories:

- **Training images:** 700 images per category
- **Held-out validation images:** 100 images per category

Therefore, each team receives:

- **Total training images:** 2,800
- **Total validation images:** 400
- **Total images:** 3,200

### 5.4 Validation Split Rule

The held-out validation images:

- Do not appear in the training split.
- Must be used only for performance evaluation.
- Must not be used for training or model tuning in a way that leaks validation information into the model.

### 5.5 Core Challenge

The main challenge is fine-grained classification between visually similar aerial-image categories.

The project should therefore pay careful attention to:

- Subtle visual differences between classes
- Class-specific spatial patterns
- Texture, shape, colour, and layout cues
- Model generalisation
- Confusion between similar categories

---

## 6. Required End-to-End Machine Learning Pipeline

Teams must implement and document a complete machine learning pipeline.

The pipeline must include the following components.

### 6.1 Problem Framing

The project must clearly explain:

- Background and motivation
- Real-world relevance of aerial scene classification
- Problem statement
- Project objectives
- Why the assigned categories are challenging

### 6.2 Exploratory Data Analysis

The project must include EDA covering the assigned image dataset.

Recommended analysis includes:

- Number of images per class
- Example images from each class
- Image dimensions and format
- Colour distribution or channel statistics
- Visual similarities and differences between classes
- Potential class ambiguity
- Data quality issues, if any
- Implications of EDA findings for preprocessing and modelling

### 6.3 Data Preparation and Preprocessing

The project must describe and justify all preprocessing steps.

Possible preprocessing steps include:

- Image resizing
- Normalisation
- Train/validation loading strategy
- Label encoding
- Batch preparation
- Dataset splitting strategy, if additional internal splits are used
- Handling corrupted or unreadable images, if applicable

### 6.4 Data Augmentation

The project should use suitable augmentation strategies to improve generalisation.

Possible augmentation techniques include:

- Random horizontal and vertical flips
- Rotation
- Random cropping or resized cropping
- Colour jitter
- Brightness and contrast changes
- Scaling
- Translation
- Other image transformations appropriate for aerial imagery

All augmentation choices must be justified in relation to the aerial-image classification task.

### 6.5 Model Development

Teams must propose and implement multiple deep learning approaches.

The number of distinct approaches investigated must be **at least equal to the number of team members**.

For a 5-member team, this means at least **5 distinct approaches** should be investigated.

Each approach should be clearly documented, including:

- Model architecture
- Whether it is trained from scratch or uses transfer learning
- Input image size
- Optimiser
- Learning rate
- Batch size
- Number of epochs
- Loss function
- Regularisation methods
- Scheduler, if used
- Any model-specific preprocessing
- Rationale for choosing the approach

### 6.6 Training and Hyperparameter Tuning

The project must include model training and tuning.

The report should document:

- Training procedure
- Hardware or runtime environment used
- Hyperparameters tested
- Tuning strategy
- Training and validation curves, where appropriate
- Overfitting or underfitting observations
- Final selected hyperparameters

### 6.7 Inference

The project must include inference on the held-out validation set.

Inference outputs should support:

- Predicted class labels
- Confidence scores or probabilities
- Per-image prediction analysis, where useful
- Identification of common misclassification cases

### 6.8 Performance Evaluation

All evaluation metrics must be clearly documented in the final report.

Required evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The evaluation should also include discussion of:

- Best-performing model
- Class-wise performance
- Misclassified examples
- Common confusion patterns
- Strengths and weaknesses of each approach
- Key insights from comparative benchmarking

---

## 7. Deep Learning Frameworks and Libraries

Teams may use modern deep learning frameworks such as:

- PyTorch
- TensorFlow
- MXNet
- Equivalent deep learning libraries

Open-source libraries are permitted, but teams must:

1. Register every library used.
2. Clearly justify the purpose of each library.
3. Document their own contributions, enhancements, and customisations.
4. Demonstrate meaningful understanding and modification of any external code or pretrained model used.

Simply downloading and executing an existing repository without meaningful understanding and modification is not acceptable.

---

## 8. Required Deep Learning Approaches

### 8.1 Minimum Number of Approaches

The project must investigate at least as many distinct deep learning approaches as there are team members.

For example:

| Team Size | Minimum Distinct Approaches |
|---:|---:|
| 5 members | 5 approaches |

### 8.2 Approach Comparison Requirements

For each approach, the report should discuss:

- Motivation for the approach
- Architecture design or selected pretrained backbone
- Training strategy
- Hyperparameters
- Performance results
- Strengths
- Weaknesses
- Failure cases
- Lessons learned

### 8.3 Comparative Analysis

The final report must present key insights from comparing the approaches.

The comparison should explain:

- Which model performed best
- Why it likely performed best
- Which models underperformed
- Whether transfer learning helped
- Whether augmentation improved generalisation
- Whether model complexity affected performance
- Practical trade-offs such as accuracy, inference speed, model size, and deployment complexity

---

## 9. Containerised Deployment Requirement

The final best-performing model must be packaged and deployed through containerisation, such as Docker.

### 9.1 Required Deployment Behaviour

The containerised application should expose an inference endpoint that can:

1. Accept an aerial image as input.
2. Run inference using the final selected model.
3. Return the predicted category label.
4. Return confidence scores.

### 9.2 Deployment Documentation

The report or supplementary materials should document:

- Container build instructions
- Container run instructions
- Inference endpoint usage
- Expected input format
- Expected output format
- Model loading process
- Any dependencies required for deployment

### 9.3 Suggested Endpoint Contract

If implementing an HTTP API, a suitable endpoint contract is:

- **Endpoint:** `POST /predict`
- **Input:** image file upload or encoded image payload
- **Output:** JSON response containing predicted label and confidence scores

Example response shape:

```json
{
  "predicted_label": "class_name",
  "confidence": 0.94,
  "scores": {
    "class_a": 0.01,
    "class_b": 0.03,
    "class_c": 0.94,
    "class_d": 0.02
  }
}
```

---

## 10. Final Deliverables

Each group must submit one package to the designated xSITE Dropbox.

The submission package must contain:

1. `T<Team Number>.pdf` — final report  
   Example: `T01.pdf`

2. `T<Team Number>.zip` — compressed file containing the video and supplementary materials  
   Example: `T01.zip`

The supplementary materials should include all files necessary to support the project submission, such as source code, model artefacts where appropriate, containerisation files, instructions, and the video demonstration.

---

## 11. Final Report Requirements

The final report should be approximately **30 pages** and must not exceed **50 pages**.

### 11.1 Required Report Structure

The report must include the following sections.

## 11.1.1 Overall Project Description

Recommended length: **3–6 pages**

Required content:

- Background and motivations
- Problem statement
- Project objectives
- Review of existing approaches

## 11.1.2 Machine Learning Solutions

Recommended length: **15–18 pages**

Required content:

- Exploratory data analysis
- Data preparation and preprocessing
- Investigation of deep learning approaches
- Model training and tuning
- Deep learning inference
- Performance evaluation
- Strengths and weaknesses
- Containerisation
- Results, insights, and discussions

## 11.1.3 Individual Contributions

Recommended length: **1–2 pages**

Required content:

- Clear and concise description of each team member's contribution
- Responsibilities for data processing, modelling, evaluation, deployment, report writing, or other project areas

## 11.1.4 Reflection

Recommended length: **2–3 pages**

Required content:

- What the team learned
- How machine learning concepts were applied to the project
- Reflections on challenges and decisions

## 11.1.5 Future Work

Recommended length: **1–2 pages**

Required content:

- Proposed future directions
- Potential model improvements
- Possible deployment improvements
- Additional experiments that could be conducted with more time or data

---

## 12. Video Demonstration Requirements

The project must include a **3–5 minute video demonstration**.

The video should demonstrate:

- Model performance
- Key findings
- Important results
- Deployment or inference functionality, where appropriate
- Concise explanation of the team's best solution

---

## 13. Assessment Criteria

The group assignment will be assessed using the following marking scheme.

Criteria may be revised based on progress and circumstances throughout the trimester.

| Assessment Component | Required Content | Weight |
|---|---|---:|
| Understanding and Solving a Real-World Challenge | Background and motivations; problem statement and project objectives; survey of existing approaches | 20% |
| Machine Learning Solutions | EDA; data preparation and preprocessing; investigation of deep learning approaches; model training and tuning; deep learning inference; performance evaluation; strengths and weaknesses; results, insights, and discussions; deployment through containerisation | 60% |
| Report Quality | Professional writing with clear logic and structure; reflection on learning and application to the project; future directions | 10% |
| Video Demonstration | 3–5 minute video demonstrating model performance and key findings | 10% |

Group marks may be weighted by peer-review scores where necessary.

Bonus marks may be awarded for:

- Exceptional depth of understanding
- Innovative techniques
- Insightful analysis beyond the baseline requirements

---

## 14. Team Requirements

- Students will be assigned to teams of 5 members in Week 3.
- Each team will receive a unique dataset partition.
- Each dataset partition contains 4 confusable aerial-image categories.
- Teams must develop their own machine learning solution independently from Week 4 onward.

---

## 15. Late Submission Policy

- A penalty of **20% per day** will be imposed for late submissions unless an extension has been approved by the lecturers before the deadline.
- Extension requests are considered on a case-by-case basis.
- Submissions received more than **4 days after the deadline** will not be accepted.
- Submissions more than 4 days late will receive zero marks.

---

## 16. Academic Integrity and Plagiarism Requirements

All submitted work must be the team's own.

The following are strictly prohibited without proper attribution:

- Copying code from another person or team
- Copying models from another person or team
- Copying written content from another person or team
- Making the team's work accessible to others for copying
- Submitting downloaded repositories without meaningful understanding, modification, and documentation

If plagiarism is detected, all involved parties will receive zero marks for the entire project.

Teams must ensure that:

- External code and libraries are properly acknowledged.
- Open-source tools are registered and justified.
- Custom contributions are clearly identified.
- All written explanations and analysis are original.

---

## 17. Practical Project Checklist

### 17.1 Dataset and EDA

- [ ] Confirm assigned 4 categories.
- [ ] Verify 700 training images per class.
- [ ] Verify 100 held-out validation images per class.
- [ ] Inspect sample images from each category.
- [ ] Analyse class similarities and likely confusion points.
- [ ] Document image dimensions, formats, and data quality.

### 17.2 Modelling

- [ ] Implement at least one approach per team member.
- [ ] Document each model architecture.
- [ ] Justify framework and library choices.
- [ ] Record all training hyperparameters.
- [ ] Track training and validation performance.
- [ ] Save model checkpoints or final model artefacts.

### 17.3 Evaluation

- [ ] Evaluate on the held-out validation set.
- [ ] Report accuracy.
- [ ] Report precision.
- [ ] Report recall.
- [ ] Report F1-score.
- [ ] Include a confusion matrix.
- [ ] Discuss class-wise results.
- [ ] Compare all investigated approaches.
- [ ] Identify the final best-performing model.

### 17.4 Deployment

- [ ] Package the final model in a container.
- [ ] Provide build instructions.
- [ ] Provide run instructions.
- [ ] Expose an inference endpoint.
- [ ] Accept an aerial image as input.
- [ ] Return predicted label and confidence scores.
- [ ] Demonstrate inference in the video or supplementary materials.

### 17.5 Report and Submission

- [ ] Write final report of approximately 30 pages.
- [ ] Ensure report does not exceed 50 pages.
- [ ] Include overall project description.
- [ ] Include machine learning solutions section.
- [ ] Include individual contributions.
- [ ] Include reflection.
- [ ] Include future work.
- [ ] Prepare 3–5 minute video demonstration.
- [ ] Package supplementary materials into `T<Team Number>.zip`.
- [ ] Submit `T<Team Number>.pdf` and `T<Team Number>.zip` to xSITE before the deadline.

---

## 18. Definition of Done

The project can be considered complete only when all of the following are true:

1. The assigned dataset has been explored, processed, and documented.
2. At least one distinct deep learning approach per team member has been implemented and evaluated.
3. The held-out validation set has been used properly for final evaluation.
4. Accuracy, precision, recall, F1-score, and confusion matrix results are reported.
5. Strengths and weaknesses of all approaches are discussed.
6. The best-performing model has been selected based on evidence.
7. The final model is deployed through a containerised inference system.
8. The inference endpoint accepts an aerial image and returns a predicted label with confidence scores.
9. The final report follows the required structure and page limits.
10. Individual contributions are clearly documented.
11. A 3–5 minute video demonstration is prepared.
12. The final submission package follows the required naming convention.
13. All external libraries, code, models, and references are properly acknowledged.
14. The submission is made before the deadline.
