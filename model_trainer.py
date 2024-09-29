
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Function to process and rank resumes
def process_and_rank_resumes(pdf_paths, hr_scores):
    
    from src.pdf_extractor import extract_text_from_pdf
    resume_texts = [extract_text_from_pdf(pdf) for pdf in pdf_paths]
    
    
    resume_embeddings = np.array([[len(text)] for text in resume_texts])
    
    
    scaler = StandardScaler()
    resume_embeddings = scaler.fit_transform(resume_embeddings)
    
    # Train and evaluate SVM models using different kernels
    kernels = ['linear', 'poly', 'rbf']
    best_kernel = None
    best_model = None
    best_mse = float('inf')
    
    for kernel in kernels:
        model = SVR(kernel=kernel)
        model.fit(resume_embeddings, hr_scores)
        predictions = model.predict(resume_embeddings)
        
        mse = mean_squared_error(hr_scores, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(hr_scores, predictions)
        
        print(f"Kernel: {kernel} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}")
        
        if mse < best_mse:  
            best_mse = mse
            best_kernel = kernel
            best_model = model  
    
    print(f"Best performing kernel: {best_kernel}")
    
    return best_model, scaler  

# Function to rank new resumes based on the best SVM model
def rank_new_resumes(new_pdf_paths, best_model, scaler):
    from src.pdf_extractor import extract_text_from_pdf
    new_resume_texts = [extract_text_from_pdf(pdf) for pdf in new_pdf_paths]
    new_resume_embeddings = np.array([[len(text)] for text in new_resume_texts])
    new_resume_embeddings = scaler.transform(new_resume_embeddings)
    
    predicted_scores = best_model.predict(new_resume_embeddings)
    
    
    ranked_resumes = sorted(zip(new_pdf_paths, predicted_scores), key=lambda x: x[1], reverse=True)
    
    return ranked_resumes
