# -------------------------------------------------------
# COMPLEX NUMBER CALCULATOR
# Author: Benjelyn Reves Patiag
# Date Created: 5-Dec-2025
# Description: This program operates on two complex numbers to perform
# addition, subtraction, multiplication, division, and modulus.
# It also displays the conjugates of the input complex number(s).
# -------------------------------------------------------


import re
import math

# ===============================================
# CLASS: Handles ONE complex number.
# ===============================================
class ComplexNumber:
    def __init__(self, real=0, imaginary=0):
        # Store real and imaginary parts
        self.real = real
        self.imaginary = imaginary

    @staticmethod
    def from_input():
        """
        Ask user for a complex number using 'i' format,like: 3+4i, 4i, 7, -3+i, etc.
        """
        while True:
            user_input = input("Enter a complex number (example: 5+7i): ").replace(" ", "")

            # Regular expression that checks if input is valid complex number format
            if re.fullmatch(r"[+-]?\d*\.?\d*([+-]\d*\.?\d*i)?", user_input) is None:
                print("Invalid format! Try something like 3+4i or -2-5i")
                continue

            # Convert "i" → "j" so Python can understand it
            python_format = user_input.replace("i", "j")

            try:
                complex_obj = complex(python_format)
                return ComplexNumber(complex_obj.real, complex_obj.imag)
            except:
                print("Invalid number! Try again.")

    def conjugate(self):
        # Conjugate: flip the sign of imaginary part
        return ComplexNumber(self.real, -self.imaginary)

    def modulus(self):
        # Magnitude formula: sqrt(a² + b²)
        return math.sqrt(self.real**2 + self.imaginary**2)

    def to_string(self):
        """
        Convert Python complex number back to 'i' format
        Removes unnecessary decimals
        """

        def clean(num):
            if num.is_integer():
                return str(int(num))
            else:
                return str(round(num, 2))

        real = clean(self.real)
        imag = clean(abs(self.imaginary))  # imaginary ALWAYS positive for display

        if self.imaginary >= 0:
            return f"{real}+{imag}i"
        else:
            return f"{real}-{imag}i"


# ===============================================
# CLASS: Performs operations on two complex numbers
# ===============================================
class ComplexCalculator:
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def add(self):
        return ComplexNumber(self.c1.real + self.c2.real,
                             self.c1.imaginary + self.c2.imaginary)

    def subtract(self):
        return ComplexNumber(self.c1.real - self.c2.real,
                             self.c1.imaginary - self.c2.imaginary)

    def multiply(self):
        real = self.c1.real * self.c2.real - self.c1.imaginary * self.c2.imaginary
        imaginary = self.c1.real * self.c2.imaginary + self.c1.imaginary * self.c2.real
        return ComplexNumber(real, imaginary)

    def divide(self):
        denominator = self.c2.real**2 + self.c2.imaginary**2
        if denominator == 0:
            return None  # Division not allowed
        real = (self.c1.real * self.c2.real + self.c1.imaginary * self.c2.imaginary) / denominator
        imaginary = (self.c1.imaginary * self.c2.real - self.c1.real * self.c2.imaginary) / denominator
        return ComplexNumber(real, imaginary)


# ===============================================
# MAIN PROGRAM
# ===============================================
def main():
    print("=== COMPLEX NUMBER CALCULATOR (using i notation) ===")

    # Get two complex numbers from the user
    print("\nEnter FIRST number: a + bi")
    c1 = ComplexNumber.from_input()

    print("\nEnter SECOND number: a + bi")
    c2 = ComplexNumber.from_input()

    # Create calculator object
    calc = ComplexCalculator(c1, c2)

    # Perform operations
    addition = calc.add()
    subtraction = calc.subtract()
    multiplication = calc.multiply()
    division = calc.divide()
    modulus1 = c1.modulus()
    modulus2 = c2.modulus()
    conjugate1 = c1.conjugate()
    conjugate2 = c2.conjugate()

    # Display results
    print("\n========== RESULTS ==========")
    print(f"Number 1: {c1.to_string()}")
    print(f"Number 2: {c2.to_string()}")

    print("\nOperations:")
    print(f"Addition:        {addition.to_string()}")
    print(f"Subtraction:     {subtraction.to_string()}")
    print(f"Multiplication:  {multiplication.to_string()}")

    if division is None:
        print("Division:        Error! Cannot divide by zero.")
    else:
        print(f"Division:        {division.to_string()}")

    print("\nExtra:")
    print(f"Modulus of #1:   {modulus1:.2f}")
    print(f"Modulus of #2:   {modulus2:.2f}")
    print(f"Conjugate #1:    {conjugate1.to_string()}")
    print(f"Conjugate #2:    {conjugate2.to_string()}")
    print("=============END=================")


# Run the program
if __name__ == "__main__":
    main()
