import fitz  # PyMuPDF for PDF text & image extraction
import re
import easyocr
from PIL import Image
import io
import numpy as np

from pix2tex.cli import LatexOCR

from image_to_text import convert_image_to_text  # Updated imports
from num2words import num2words

# # Configure Tesseract path (update this based on your system)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# # Initialize EasyOCR reader once
# reader = easyocr.Reader(['en'])
latex_ocr = LatexOCR()

def convert_numbers_to_words(match):
    num_str = match.group()
    try:
        return num2words(int(num_str))
    except ValueError:
        return num_str  # fallback if conversion fails

class MathToSpeech:
    def __init__(self):
        # Define replacements for LaTeX symbols with properly escaped patterns
        self.latex_replacements = {
            # Basic operations
            r"\\times\b": " times ",
            r"\\cdot\b": " dot ",
            r"\\div\b": " divided by ",

            # Fractions and division - properly escape curly braces
            r"\\frac\{(.*?)\}\{(.*?)\}": r"\1 over \2",
            r"\\dfrac\{(.*?)\}\{(.*?)\}": r"\1 over \2",
            r"\\tfrac\{(.*?)\}\{(.*?)\}": r"\1 over \2",

            # Exponents and subscripts
            r"(\w)\^(\{?)([^\}]+)(\}?)": r"\1 to the power of \3",
            r"(\w)_(\{?)([^\}]+)(\}?)": r"\1 subscript \3",
            r"\\exp\b": "exponential of",

            # Roots - properly escape square brackets and curly braces
            r"\\sqrt\[(.*?)\]\{(.*?)\}": r"\1-th root of \2",
            r"\\sqrt\{(.*?)\}": r"square root of \1",
            r"\\sqrt\b": "square root of",

            # Summation and products - properly escape curly braces
            r"\\sum\_\{(.*?)\}\^\{(.*?)\}": r"summation from \1 to \2 of",
            r"\\sum\_\{(.*?)\}": r"summation over \1 of",
            r"\\sum\b": "summation of",
            r"\\prod\_\{(.*?)\}\^\{(.*?)\}": r"product from \1 to \2 of",
            r"\\prod\_\{(.*?)\}": r"product over \1 of",
            r"\\prod\b": "product of",

            # Integrals
            r"\\int\_\{(.*?)\}\^\{(.*?)\}": r"integral from \1 to \2 of",
            r"\\int\_\{(.*?)\}": r"integral over \1 of",
            r"\\int\b": "integral of",
            r"\\iint\b": "double integral of",
            r"\\iiint\b": "triple integral of",
            r"\\oint\b": "contour integral of",

            # Limits
            r"\\lim\_\{(.*?)\}": r"limit as \1 of",

            # Matrices - properly escape curly braces
            r"\\begin\{matrix\}(.*?)\\end\{matrix\}": r"matrix \1",
            r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}": r"parenthesis matrix \1",
            r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}": r"bracket matrix \1",

            # Greek letters and other symbols
            r"\\alpha\b": "alpha",
            r"\\beta\b": "beta",
            r"\\gamma\b": "gamma",
            r"\\delta\b": "delta",
            r"\\epsilon\b": "epsilon",
            r"\\zeta\b": "zeta",
            r"\\eta\b": "eta",
            r"\\theta\b": "theta",
            r"\\iota\b": "iota",
            r"\\kappa\b": "kappa",
            r"\\lambda\b": "lambda",
            r"\\mu\b": "mu",
            r"\\nu\b": "nu",
            r"\\xi\b": "xi",
            r"\\pi\b": "pi",
            r"\\rho\b": "rho",
            r"\\sigma\b": "sigma",
            r"\\tau\b": "tau",
            r"\\upsilon\b": "upsilon",
            r"\\phi\b": "phi",
            r"\\chi\b": "chi",
            r"\\psi\b": "psi",
            r"\\omega\b": "omega",
            r"\\Gamma\b": "Gamma",
            r"\\Delta\b": "Delta",
            r"\\Theta\b": "Theta",
            r"\\Lambda\b": "Lambda",
            r"\\Xi\b": "Xi",
            r"\\Pi\b": "Pi",
            r"\\Sigma\b": "Sigma",
            r"\\Upsilon\b": "Upsilon",
            r"\\Phi\b": "Phi",
            r"\\Psi\b": "Psi",
            r"\\Omega\b": "Omega",

            # Other symbols
            r"\\infty\b": "infinity",
            r"\\pm\b": "plus or minus",
            r"\\mp\b": "minus or plus",
            r"\\approx\b": "approximately equal to",
            r"\\sim\b": "similar to",
            r"\\cong\b": "congruent to",
            r"\\equiv\b": "equivalent to",
            r"\\propto\b": "proportional to",
            r"\\parallel\b": "parallel to",
            r"\\perp\b": "perpendicular to",
            r"\\angle\b": "angle",
            r"\\degree\b": "degrees",
            r"\\circ\b": "circle",
            r"\\prime\b": "prime",
            r"\\hbar\b": "h bar",
            r"\\ell\b": "script l",

            # Functions
            r"\\sin\b": "sine of",
            r"\\cos\b": "cosine of",
            r"\\tan\b": "tangent of",
            r"\\sec\b": "secant of",
            r"\\csc\b": "cosecant of",
            r"\\cot\b": "cotangent of",
            r"\\arcsin\b": "arc sine of",
            r"\\arccos\b": "arc cosine of",
            r"\\arctan\b": "arc tangent of",
            r"\\sinh\b": "hyperbolic sine of",
            r"\\cosh\b": "hyperbolic cosine of",
            r"\\tanh\b": "hyperbolic tangent of",
            r"\\log\b": "logarithm of",
            r"\\ln\b": "natural logarithm of",
            r"\\lg\b": "logarithm base 10 of",
        }

        # Text equation replacements (remain unchanged)
        self.text_replacements = {
            "=": " equals ",
            "+": " plus ",
            "-": " minus ",
            "*": " times ",
            "/": " divided by ",
            "^": " to the power of ",
            "(": " open parenthesis ",
            ")": " close parenthesis ",
            "[": " open bracket ",
            "]": " close bracket ",
            "{": " open brace ",
            "}": " close brace ",
            "<": " less than ",
            ">": " greater than ",
            "≤": " less than or equal to ",
            "≥": " greater than or equal to ",
            "≠": " not equal to ",
            "≈": " approximately equal to ",
            "≡": " equivalent to ",
            "∝": " proportional to ",
            "√": " square root of ",
            "∑": " summation of ",
            "∏": " product of ",
            "∫": " integral of ",
            "∂": " partial derivative of ",
            "∇": " nabla ",
            "∞": " infinity ",
            "π": " pi ",
            "θ": " theta ",
            "α": " alpha ",
            "β": " beta ",
            "γ": " gamma ",
            "δ": " delta ",
            "ε": " epsilon ",
            "λ": " lambda ",
            "μ": " mu ",
            "σ": " sigma ",
            "ω": " omega ",
            "∈": " in ",
            "∉": " not in ",
            "⊂": " subset of ",
            "⊆": " subset of or equal to ",
            "∪": " union ",
            "∩": " intersect ",
            "∀": " for all ",
            "∃": " there exists ",
            "¬": " not ",
            "∧": " and ",
            "∨": " or ",
            "⇒": " implies ",
            "⇔": " if and only if ",
            "⊥": " perpendicular to ",
            "∥": " parallel to ",
            "∠": " angle ",
            "°": " degrees ",
            "'": " prime ",
            "→": " approaches ",
            "←": " maps to ",
            "↔": " if and only if ",
            "↑": " up arrow ",
            "↓": " down arrow ",
            "↗": " north east arrow ",
            "↘": " south east arrow ",
            "↖": " north west arrow ",
            "↙": " south west arrow ",
            "∴": " therefore ",
            "∵": " because ",
            "ℵ": " aleph ",
            "ℏ": " h bar ",
            "ℜ": " real part of ",
            "ℑ": " imaginary part of ",
            "ℂ": " complex numbers ",
            "ℝ": " real numbers ",
            "ℚ": " rational numbers ",
            "ℤ": " integers ",
            "ℕ": " natural numbers ",
            "∅": " empty set ",
            "⊕": " direct sum ",
            "⊗": " tensor product ",
            "⊙": " circled dot ",
            "†": " dagger ",
            "‡": " double dagger ",
            "§": " section ",
            "¶": " paragraph ",
            "…": " ellipsis ",
            "⋯": " midline ellipsis ",
            "⋮": " vertical ellipsis ",
            "⋱": " diagonal ellipsis ",
            "⋰": " up diagonal ellipsis ",
        }

        # OCR corrections (remain unchanged)
        self.ocr_corrections = {
            "?": "^",  # Sometimes `^` is misread
            "×": "*",  # Proper multiplication symbol
            "−": "-",  # Correct minus sign
            "—": "-",  # Long dash misread as minus
            "÷": "/",  # Proper division symbol
            "‘": "'",  # Smart quotes
            "’": "'",
            "“": '"',
            "”": '"',
            "¦": "|",  # Broken bar
            "¬": "-",  # Sometimes misread as negative sign
            "£": "#",  # Sometimes misread as number sign
            "€": "∈",  # Sometimes misread as element symbol
            "§": "5",  # Sometimes misread as 5
            "©": "(",  # Sometimes misread as parenthesis
            "®": ")",  # Sometimes misread as parenthesis
            "™": "+",  # Sometimes misread as plus
            "¢": "c",  # Sometimes misread as c
            "¥": "y",  # Sometimes misread as y
            "µ": "μ",  # Micro sign to mu
            "º": "°",  # Ordinal indicator to degree
            "ª": "a",  # Feminine ordinal indicator to a
            "¿": "?",  # Inverted question mark
            "¡": "!",  # Inverted exclamation mark
            "«": "<<",  # Left angle quotes
            "»": ">>",  # Right angle quotes
        }

    def process_latex_equation(self, latex_code):
        """Convert LaTeX equation to spoken English."""
        # Remove LaTeX math mode delimiters
        latex_code = re.sub(r"^\$|\$$", "", latex_code)
        latex_code = re.sub(r"^\\\(|\\\)$", "", latex_code)
        latex_code = re.sub(r"^\\\[|\\\]$", "", latex_code)

        # Process each replacement pattern
        for pattern, replacement in self.latex_replacements.items():
            try:
                latex_code = re.sub(pattern, replacement, latex_code)
            except re.error as e:
                print(f"Error in pattern: {pattern}")
                print(f"Error message: {str(e)}")
                continue

        # Clean up spaces
        latex_code = re.sub(r"\s+", " ", latex_code).strip()

        return latex_code


    def process_text_equation(self, text):
        """Convert plain text equation to spoken English."""
        text = text.strip()

        # Replace each operator in the text
        for symbol, spoken in self.text_replacements.items():
            text = text.replace(symbol, spoken)

        # Clean up spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def correct_ocr_errors(self, text):
        """Fix common OCR misinterpretations."""
        for wrong, correct in self.ocr_corrections.items():
            text = text.replace(wrong, correct)
        return text

    def is_valid_latex(self, text):
        if not text or len(text) > 200:
            return False
        if text.count('(') != text.count(')') or text.count('{') != text.count('}'):
            return False
        if not re.search(r"\\[a-zA-Z]+", text):  # No LaTeX commands
            return False
        if not any(op in text for op in ['+', '-', '=', '\\frac', '\\sum', '\\int']):  # No important math stuff
            return False
        # New: Reject if too many spaces (broken OCRs are full of spaces)
        if text.count(' ') > len(text) * 0.3:
            return False
        # New: Reject if too many single characters separated by spaces
        if re.search(r'(\\[a-z])(\s)', text):  # Like \s \c \r
            return False
        return True


    
    def process_image(self, image):
        extracted_text = latex_ocr(image)
        equation_text = " ".join(extracted_text).strip()
        equation_text = self.correct_ocr_errors(equation_text)

        if self.is_valid_latex(equation_text):
            return self.process_latex_equation(equation_text)
        elif any(op in equation_text for op in ['+', '-', '*', '/', '=', '^', '<', '>', '≤', '≥', '≠', '≈', '∑', '∫', '∏']) and len(equation_text) < 150 and not equation_text.startswith("\\"):
            return self.process_text_equation(equation_text)
        else:
            return convert_image_to_text(image)


    
    def process_numbers(self,text):
        text = re.sub(r'\b\d+\b', convert_numbers_to_words, text)
        return text


    # def extract_equations_from_pdf(self, pdf_path):
    #     """Scan a PDF and categorize equations."""
    #     doc = fitz.open(pdf_path)
    #     results = []

    #     for page_num, page in enumerate(doc):
    #         text = page.get_text("text")

    #         # Detect LaTeX equations in text
    #         latex_equations = re.findall(r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", text)
    #         for eq in latex_equations:
    #             results.append({
    #                 "type": "latex",
    #                 "equation": eq,
    #                 "spoken": self.process_latex_equation(eq),
    #                 "page": page_num + 1
    #             })

           
    #         for eq in text_equations:
    #             if any(op in eq for op in ['+', '-', '*', '/', '=', '^', '<', '>', '≤', '≥', '≠', '≈', '∑', '∫', '∏']):
    #                 results.append({
    #                     "type": "text",
    #                     "equation": eq,
    #                     "spoken": self.process_text_equation(eq),
    #                     "page": page_num + 1
    #                 })

    #         # Extract images and check for equation presence
    #         for img_index, img in enumerate(page.get_images(full=True)):
    #             xref = img[0]
    #             base_image = doc.extract_image(xref)
    #             image_bytes = base_image["image"]
    #             image = Image.open(io.BytesIO(image_bytes))

    #             # Use OCR to detect if the image contains an equation
    #             extracted_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
    #             extracted_text = self.correct_ocr_errors(extracted_text)

    #             if extracted_text and any(char.isdigit() or char in "+-*/=^<>∑∫∏∂∇" for char in extracted_text):
    #                 results.append({
    #                     "type": "image",
    #                     "equation": extracted_text,
    #                     "spoken": self.process_image_equation(image),
    #                     "page": page_num + 1,
    #                     "image_index": img_index + 1
    #                 })

    #     return results

# # Example Usage
# if __name__ == "__main__":
#     converter = MathToSpeech()

#     # Test LaTeX equations
#     latex_examples = [
#         r"$\sum_{i=1}^n i = \frac{n(n+1)}{2}$",  # Summation formula
#         r"$\frac{d}{dx}e^x = e^x$",  # Derivative
#         r"$\int_0^1 x^2 dx = \frac{1}{3}$",  # Integral
#         r"$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$",  # Maxwell's equation
#         r"$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$",  # Matrix
#         (r"$f(x) = \begin{cases} "
#         r"1 & \text{if } x \in \mathbb{Q} \\ "
#         r"0 & \text{if } x \notin \mathbb{Q} "
#         r"\end{cases}$")  # Piecewise function with concatenation
#     ]

#     print("LaTeX Examples:")
#     for eq in latex_examples:
#         print(f"Input: {eq}")
#         print(converter.process_latex_equation(eq))
#         print()

#     # Test text equations
#     text_examples = [
#         "3x^2 + 2x - 5 = 0",
#         "A = πr²",
#         "∑(1/n^2) = π²/6",
#         "∫e^x dx = e^x + C",
#         "∇×E = -∂B/∂t",
#         "∀ε>0, ∃δ>0 s.t. |x-a|<δ ⇒ |f(x)-L|<ε"
#     ]

#     print("\nText Examples:")
#     for eq in text_examples:
#         print(f"Input: {eq}")
#         print(converter.process_text_equation(eq))
#         print()


