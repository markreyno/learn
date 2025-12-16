
    

questions = {"What is the capital of France?": "Paris"
            , "What is the capital of Germany?": "Berlin"
            , "What is the capital of Italy?": "Rome"
            , "What is the capital of Spain?": "Madrid"
            , "What is the capital of Portugal?": "Lisbon"
            , "What is the capital of Greece?": "Athens"
            , "What is the capital of Turkey?": "Ankara"
            , "What is the capital of Egypt?": "Cairo"
            , "What is the capital of Nigeria?": "Abuja"}
   
def ask_questions():
    score = 0
    for question, answer in questions.items():
        user_answer = input(question + " ")
        
        if user_answer.lower() == answer.lower():
            print("Correct!")
            score += 1
        else:
            print(f"Incorrect! The correct answer is {answer}")
        
    print(f"End of quiz. Your score is {score}/{len(questions)}")
    
    


def main():
    
    
    start = input("Welcome the captial quiz, are you ready to start? (y/n):")
    
    if start == "y":
        ask_questions()
    else:
        print("Thank you for playing the capital quiz. Goodbye!")
        
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()