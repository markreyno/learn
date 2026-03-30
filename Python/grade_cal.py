def Welcome():
    ("Welcome to the grade calculator")
    print("--------------------------------")
    
def check_score_values():







    
def Add_assignment(assignments):
    name = input("enter the name of the assignment:")
    
  
    
    
    
    
    
    max_score = int(input("enter the max score of the assignment:"))
    type = input("enter the type of the assignment: ")
    assignments.append({"name": name, "score": score, "max_score": max_score, "percentage_weight": type})
    print(f"Assignment '{name}' added to the list")

def View_assignments(assignments):
    pass

def Calculate_total_grade(assignments):
    pass



    
def main():
    
    assignments = [
        {"name": "Quiz 1", "score": 80, "max_score": 100, "percentage_weight": "quiz"},
        {"name": "Quiz 2", "score": 90, "max_score": 100, "percentage_weight": "quiz"},
        {"name": "Assignment 1", "score": 70, "max_score": 100, "percentage_weight": "assignment"},
        {"name": "Homework 1", "score": 85, "max_score": 100, "percentage_weight": "homework"},
        {"name": "Exam 1", "score": 95, "max_score": 100, "percentage_weight": "exam"},
    ]
    Welcome()
    
    while True:
        print("Select an option:")
        print("1. Add Assignment")
        print("2. View Assignments")
        print("3. Calculate Total Grade")
        print("4. Exit")
        
        choice = int(input("Enter your choice: "))
        
        if choice == 1:
            Add_assignment(assignments)
        elif choice == 2:
            View_assignments(assignments)
        elif choice == 3:
            Calculate_total_grade(assignments)
        elif choice == 4:
            break
        else:
            print("Invalid option. Please try again.")
    
    
    
if __name__ == "__main__":
    main()
    