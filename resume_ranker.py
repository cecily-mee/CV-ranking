

from src.model_trainer import process_and_rank_resumes, rank_new_resumes

def main():
    num_resumes = int(input("Enter number of resumes: "))
    resume_pdfs = input(f"Enter the names of {num_resumes} resume PDFs (comma-separated): ").split(",")
    resume_pdfs = [pdf.strip() for pdf in resume_pdfs]  # Clean up input
    
    hr_scores = []
    for pdf in resume_pdfs:
        score = float(input(f"Enter HR score for resume {pdf}: "))
        hr_scores.append(score)
    
    # Process the resumes and train SVM models
    best_model, scaler = process_and_rank_resumes(resume_pdfs, hr_scores)
    
    
    num_new_resumes = int(input("Enter number of new resumes to rank: "))
    new_resume_pdfs = input(f"Enter the names of {num_new_resumes} new resume PDFs (comma-separated): ").split(",")
    new_resume_pdfs = [pdf.strip() for pdf in new_resume_pdfs]
    
    
    ranked_new_resumes = rank_new_resumes(new_resume_pdfs, best_model, scaler)
    
    
    print("\nRanked New Resumes (Higher means higher priority):")
    for i, (pdf, score) in enumerate(ranked_new_resumes, 1):
        print(f"{i}. {pdf} - Predicted Score: {score:.2f}")

