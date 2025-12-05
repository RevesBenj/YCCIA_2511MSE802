# -------------------------------------------------------
# COMPLEX NUMBER CALCULATOR
# Author: Benjelyn Reves Patiag
# Date Created: 5-Dec-2025
# Description:  This program takes a complex vector and displays its phase difference and magnitude, and vice versa.
# It also converts a complex number into theta and magnitude, and converts magnitude and theta back into a complex number.
# -------------------------------------------------------



import cmath
import math


# ======================================
# CLASS: Handles ONE complex number
# ======================================
class ComplexVector:
    """Represents one complex number and all conversions/operations."""

    def __init__(self, number: complex):
        self.z = number

    @classmethod
    def from_string(cls, s: str):
        """Convert 'a + bi' into Python complex number."""
        try:
            s = s.replace(" ", "").replace("i", "j")
            return cls(complex(s))
        except Exception:
            raise ValueError("Invalid format! Use a + bi")

    @classmethod
    def from_polar(cls, magnitude: float, angle_deg: float):
        angle_rad = math.radians(angle_deg)
        z = cmath.rect(magnitude, angle_rad)
        return cls(z)

    def magnitude(self):
        return round(abs(self.z), 2)

    def angle_deg(self):
        return round(math.degrees(cmath.phase(self.z)), 2)

    def conjugate(self):
        return ComplexVector(self.z.conjugate())

    # --- Convert to a+bi string ---
    def to_complex_string(self):
        real = round(self.z.real, 2)
        imag = round(self.z.imag, 2)
        sign = "+" if imag >= 0 else "-"
        return f"{real} {sign} {abs(imag)}i"

    # --- Convert to PHASOR notation ---
    def to_phasor(self):
        return f"{self.magnitude()} ∠ {self.angle_deg()}°"

    # ===== Arithmetic operations =====
    def add(self, other):
        return ComplexVector(self.z + other.z)

    def subtract(self, other):
        return ComplexVector(self.z - other.z)

    def multiply(self, other):
        return ComplexVector(self.z * other.z)

    def divide(self, other):
        if other.z == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return ComplexVector(self.z / other.z)


# ======================================
# CLASS: Calculator for 2 vectors
# ======================================
class VectorCalculator:
    @staticmethod
    def phase_difference(v1: ComplexVector, v2: ComplexVector):
        return round(v1.angle_deg() - v2.angle_deg(), 2)


# ======================================
# MAIN PROGRAM
# ======================================
def main():
    print("\n===== COMPLEX VECTOR CALCULATOR =====")

    try:
        # ================================
        # FIRST COMPLEX NUMBER
        # ================================
        c1 = input("Enter first complex number (a + bi): ")
        v1 = ComplexVector.from_string(c1)

        print("\n--- 1. COMPLEX NUMBER/RECTANGULAR → PHASOR ---")
        print(f"Complex: {v1.to_complex_string()}")
        print(f"Phasor form: {v1.to_phasor()}")
        print(f"Conjugate: {v1.conjugate().to_complex_string()}")

        print("\n--- 1. PHASOR → COMPLEX NUMBER/RECTANGULAR ---")
        mag1 = float(input("Enter magnitude: "))
        ang1 = float(input("Enter angle (degrees): "))
        v1p = ComplexVector.from_polar(mag1, ang1)
        print(f"Result: {v1p.to_complex_string()}")

        # ================================
        # SECOND COMPLEX NUMBER (optional)
        # ================================
        c2 = input("\nEnter second complex number (ENTER to skip): ")

        if c2.strip():
            v2 = ComplexVector.from_string(c2)

            print("\n--- 2. COMPLEX NUMBER/RECTANGULAR → PHASOR ---")
            print(f"Complex: {v2.to_complex_string()}")
            print(f"Phasor form: {v2.to_phasor()}")
            print(f"Conjugate: {v2.conjugate().to_complex_string()}")

            print("\n--- 2. PHASOR → COMPLEX NUMBER/RECTANGULAR ---")
            mag2 = float(input("Enter magnitude: "))
            ang2 = float(input("Enter angle (degrees): "))
            v2p = ComplexVector.from_polar(mag2, ang2)
            print(f"Result: {v2p.to_complex_string()}")

            print("\nPhase Difference & Magnitude:")
            print(f"• Phase Difference: {VectorCalculator.phase_difference(v1, v2)}°")
            print(f"• Magnitude of first:  {v1.magnitude()}")
            print(f"• Magnitude of second: {v2.magnitude()}")

            # ================================
            # OPERATIONS
            # ================================
            print("\n===== OPERATIONS BETWEEN TWO NUMBERS =====")
            print(f"1) Addition:        {v1.add(v2).to_complex_string()}")
            print(f"2) Subtraction:     {v1.subtract(v2).to_complex_string()}")
            print(f"3) Multiplication:  {v1.multiply(v2).to_complex_string()}")
            print(f"4) Division:        {v1.divide(v2).to_complex_string()}")

            print("\nConjugates:")
            print(f"• Conjugate of first:  {v1.conjugate().to_complex_string()}")
            print(f"• Conjugate of second: {v2.conjugate().to_complex_string()}")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
