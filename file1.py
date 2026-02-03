"""
Step 1: Questions for each dataset 

# College Completion Datasets:
  - How much does tuition cost influence the completion rate of colleges
  - Does the accpetance rate(college difficulty) have a big effect on college completion? 
  - Are completion outcomes different by region/state, public vs private, or 2-year vs 4-year?
  Main Question: What characters have the most association with higher college completion rates (tuition cost, size, region, public/private)? 

# Job Placement Datasets: 
  - How much weight do internships/experience carry compared to GPA
  - Does age matter when it comes to job placement 
  - What predicts salary among those placed?
  Main Question: Which characteristics best predict whether a student gets a job? 

  Step 2: 
  # College Completion 

  - Generic question: What characters have the most association with higher 
  college completion rates (tuition cost, size, region, public/private)? 
  - Independent Business Metric: The cost to complete college
  - Variable type/class: 
      - completion rate and other continuous columns are numeric (float)
      - categorical columns (e.g., control type, region, state, level) are strings
      - integer-coded categories to strings if needed (e.g., “1/2/3” codes)
  - Collapse factor levels: 
  - One-hot encode factor variables: Apply one-hot encoding to categorical variables after collapsing
  - Normalize continuous variables: Standardize continuous predictors
  - Drop unneeded variables: Drop identifiers / non-predictive fields
  - Create target variable: Schools that are high-performing 
  student completion rate is above average completion rate)
  high_grad_150 = 1 if grad_150_value > median(grad_150_value) else 0
  - Calculate prevalence of target: Prevalence = percent of institutions labeled “high completion”
  - Create necessary partitions (Train, Tune, Test): Train: 60%, Tune/Validation: 20%, Test: 20%

  # Job Placement: 
   - Generic question: Which characteristics best predict whether a student has a salary above the median?  
  - Independent Business Metric: Salary rate  
  - Variable type/class: 
      - Convert numeric fields: GPA, test scores, experience years, salary → numeric
      - Convert categorical fields: gender, degree stream, specialization, board type → category/string
      - Ensure binary flags are 0/1 integers if present
  - Collapse factor levels: Collapse rare specializations/streams into "Other" if many unique values
  - One-hot encode factor variables: One-hot encode categorical predictors after cleaning/collapsing
  - Normalize continuous variables: Standardize continuous predictors
  - Drop unneeded variables: Drop identifiers like name, email, student ID
  - Create target variable: Median salary 
  Define:
above_median_salary = 1 if salary > median_salary_train
above_median_salary = 0 if salary <= median_salary_train
  - Calculate prevalence of target: prevalence = mean(above_median_salary)
  - Create necessary partitions (Train, Tune, Test): Train: 60%, Tune/Validation: 20%, Test: 20%

  #Step 3: What do your instincts tell you about the data. 
  Can it address your problem, what areas/items are you worried about?

  # College Completion: I think the dataset is great for addressing the main question as a result of its data including many school-level characteristics (enrollment size, public/private, region, etc.). 
    This dataset is optimal for identifying which characteristics are correlated with above-average completion, but the results should be framed as predictive associations and that all values are standardized to the same measurements. 
  # Job Placement: The job placement dataset can be used to address the problem, as the correlation between salary and the type of student one person was is easy to see. 
  However, there is a selection bias in the selection of only those people who actually secured a job, which could neglect those with similar characteristics who have jobs that are not being recorded in the data.