import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
import pytesseract
import pdfplumber
import re
import json
import time
from datetime import timedelta
import io

# --- Configure Tesseract Path (if necessary) ---
# If tesseract is not in your PATH, you might need to uncomment and set this:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Example for macOS
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows

# --- Configure Gemini API ---
# Use st.secrets for secure API key handling
# Ensure you have a .streamlit/secrets.toml file with GOOGLE_API_KEY="YOUR_API_KEY"
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"API Key configuration failed: {e}")
    st.info("Please ensure you have a `.streamlit/secrets.toml` file in your app's directory with your `GOOGLE_API_KEY`.")
    st.stop() # Stop the app if API key is not configured

# Set up the model
# gemini-pro has a decent context window for moderate amounts of text.
# gemini-1.5-flash-latest has a massive context window if you expect very long inputs.
model_name = "gemini-1.5-flash-latest" # Or "gemini-1.5-flash-latest" if preferred and available

generation_config = {
  "temperature": 0.2, # Keep temperature low for factual answers and consistent JSON
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192, # Maximum output length
  # "response_mime_type": "application/json", # Optional for 1.5 models and may require model adjustments
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

try:
    # Check if the model exists before trying to get it
    # models = [m.name for m in genai.list_models()] # Can be slow
    # if model_name not in models:
    #     st.error(f"Model '{model_name}' not found. Available models: {models}")
    #     st.stop()

    model = genai.GenerativeModel(model_name=model_name,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
except Exception as e:
    st.error(f"Failed to initialize Gemini model '{model_name}': {e}")
    st.info("Please check your API key, model name, and internet connection.")
    st.stop()


# --- Session State Initialization ---
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = 'not_started' # 'not_started', 'in_progress', 'completed'
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'time_limit' not in st.session_state:
    st.session_state.time_limit = 10 # Default minutes
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'end_time' not in st.session_state:
    st.session_state.end_time = None
if 'start_limit' not in st.session_state: # Store limit in seconds for countdown
    st.session_state.start_limit = None


# --- Core Processing Function using Gemini ---

def process_text_with_gemini_for_quiz(raw_text):
    """
    Sends raw text to Gemini to parse, find correct answers, and generate explanations.
    Expects Gemini to return a JSON array of question objects.
    """
    if not raw_text or not raw_text.strip():
        st.warning("No text provided to process.")
        return []

    # It's generally better to keep prompts focused on the task.
    # Asking for strict JSON output can sometimes be challenging for the model
    # depending on the complexity of the input text.
    # The current prompt asks for JSON and includes validation.
    # If you find it struggles with JSON output, you might try asking for a different
    # structured format first and then parsing that, or using a model with better JSON support.

    prompt = f"""Analyze the following text which contains one or more multiple-choice questions.
For each multiple-choice question found:
1. Extract the full question text.
2. Extract all options with their corresponding letters (e.g., A, B, C, D).
3. Determine the SINGLE correct answer letter (e.g., A, B, C, D) based on standard knowledge.
4. Provide a concise explanation for why the determined correct answer is correct.
5. Provide concise explanations for why each INCORRECT option is incorrect.

Structure the output as a JSON array of question objects. Each object in the array must have the following keys:
- "question_text": The text of the question.
- "options": A JSON object where keys are the option letters (A, B, C, ...) and values are the option texts.
- "correct_answer": The letter of the correct option (e.g., "A").
- "explanations": A JSON object where keys are the option letters (A, B, C, ...) and values are the explanations for that option. Ensure explanations are provided for ALL extracted options.

If no multiple-choice questions are found, return an empty JSON array `[]`.

Input Text:
---
{raw_text[:15000]} # Limit input text size for safety/token limits - adjust if needed
---

JSON Output:
""" # The prompt ends expecting a JSON array directly

    st.info(f"Sending text to Gemini for parsing and analysis ({len(raw_text)} characters)...")

    try:
        # Use the generative model
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Clean the response text to ensure it's valid JSON (remove markdown code block)
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):]
        if response_text.endswith("```"):
            response_text = response_text[:-len("```")]
        response_text = response_text.strip()

        # Attempt to parse the JSON response
        parsed_questions = json.loads(response_text)

        # Validate the parsed structure - ensure it's a list of dicts with required keys
        if not isinstance(parsed_questions, list):
             st.error("Gemini response is not a JSON array.")
             st.text("Raw response from Gemini:")
             st.text(response_text) # Show raw response
             return []

        valid_questions = []
        for i, q_data in enumerate(parsed_questions):
            # Perform basic validation for each question object
            if (isinstance(q_data, dict) and
                'question_text' in q_data and isinstance(q_data['question_text'], str) and
                'options' in q_data and isinstance(q_data['options'], dict) and
                'correct_answer' in q_data and isinstance(q_data['correct_answer'], str) and
                'explanations' in q_data and isinstance(q_data['explanations'], dict)):

                 # Further validation: check if correct_answer is one of the option letters and options are not empty
                 correct_ans_letter = q_data['correct_answer'].upper()
                 if not q_data['options']:
                     st.warning(f"Gemini returned Q{i+1} with no options.")
                     st.json(q_data)
                     continue # Skip this question
                 elif correct_ans_letter in q_data['options']:
                     # Ensure options are sorted alphabetically by key for consistency later
                     q_data['options'] = dict(sorted(q_data['options'].items()))
                     # Ensure explanations only exist for provided options and are sorted
                     q_data['explanations'] = {
                         letter: q_data['explanations'].get(letter, "No explanation provided by AI.")
                         for letter in q_data['options'].keys() # Filter and order explanations by options
                     }
                     valid_questions.append(q_data)
                 else:
                     st.warning(f"Gemini returned invalid correct answer '{correct_ans_letter}' for Q{i+1}.")
                     st.json(q_data) # Show the problematic question data
            else:
                st.warning(f"Gemini returned Q{i+1} with an invalid structure.")
                st.json(q_data) # Show the problematic data


        st.success(f"Gemini successfully processed and extracted {len(valid_questions)} valid questions.")
        if len(parsed_questions) > len(valid_questions):
             st.warning(f"Skipped {len(parsed_questions) - len(valid_questions)} questions returned by Gemini due to validation errors.")

        return valid_questions

    except json.JSONDecodeError:
         st.error("Failed to parse JSON response from Gemini. Check the raw response below.")
         st.text("Raw response from Gemini:")
         st.text(response_text) # Show raw response text
         return []
    except Exception as e:
        st.error(f"Error processing text with Gemini: {e}")
        # Optionally show response_text here if available before the error
        # st.text(response_text if 'response_text' in locals() else "No response text available.")
        return []

def reset_quiz():
    """Resets the quiz state."""
    st.session_state.quiz_state = 'not_started'
    st.session_state.user_answers = {}
    st.session_state.score = 0
    st.session_state.start_time = None
    st.session_state.end_time = None
    st.session_state.start_limit = None


# --- UI Layout ---
st.title("Gemini-Powered Quiz App")

# Sidebar Navigation
with st.sidebar:
    st.header("Navigation")
    # Added "Review Questions" to the list
    page = st.radio("Go to", ["Add Questions", "Review Questions", "Take Quiz", "View Results"])

    st.header("Manage Questions")
    if st.button("Clear All Questions"):
        st.session_state.questions = []
        reset_quiz() # Reset quiz if questions are cleared
        st.success("All questions cleared.")
        st.rerun() # Corrected from experimental_rerun


# --- Add Questions Page ---
if page == "Add Questions":
    st.header("Add Questions")

    st.write("""
    Add multiple-choice questions here by typing, uploading an image (PNG, JPG), or uploading a PDF.
    Gemini will attempt to find and structure the questions and answers from the text.
    """)

    text_input = st.text_area("Enter questions (optional):", height=150)
    image_input = st.file_uploader("Upload an image with questions (optional):", type=['png', 'jpg', 'jpeg'])
    pdf_input = st.file_uploader("Upload a PDF with questions (optional):", type=['pdf'])


    if st.button("Process and Add Questions"):
        if not text_input and not image_input and not pdf_input:
            st.warning("Please enter text, upload an image, or upload a PDF.")
        else:
            raw_text = ""

            if text_input:
                raw_text += text_input.strip() + "\n\n" # Add text area content

            if image_input:
                try:
                    st.info("Processing image...")
                    img = Image.open(image_input)
                    # st.image(img, caption="Uploaded Image", width=200) # Optional: display image
                    # Perform OCR
                    st.info("Performing OCR on image...")
                    # Add error handling for pytesseract
                    try:
                         image_text = pytesseract.image_to_string(img)
                         raw_text += image_text.strip() + "\n\n"
                         st.text_area("Text extracted from image (for review):", image_text, height=150, key="image_text_review")
                    except pytesseract.TesseractNotFoundError:
                         st.error("Tesseract is not installed or not in your PATH. Image OCR disabled.")
                    except Exception as e:
                         st.error(f"Error during Tesseract OCR: {e}")

                except Exception as e:
                    st.error(f"Error processing image file: {e}")
                    image_input = None # Prevent further processing if image fails

            if pdf_input:
                 try:
                     st.info("Processing PDF...")
                     # Read PDF as bytes, then use io.BytesIO to make it file-like
                     pdf_bytes = pdf_input.getvalue()
                     pdf_file = io.BytesIO(pdf_bytes)
                     pdf_text = ""
                     with pdfplumber.open(pdf_file) as pdf:
                         st.info(f"Extracting text from {len(pdf.pages)} pages...")
                         for page in pdf.pages:
                             pdf_text += page.extract_text() or "" # Add text, handle None if page is empty
                             pdf_text += "\n\n" # Add separation between pages
                     raw_text += pdf_text.strip() + "\n\n"
                     st.text_area("Text extracted from PDF (for review):", pdf_text, height=150, key="pdf_text_review")
                 except Exception as e:
                     st.error(f"Error processing PDF: {e}")
                     pdf_input = None # Prevent further processing if PDF fails


            if raw_text.strip():
                st.info("Combined raw text collected. Sending to Gemini for parsing and analysis...")
                # Use Gemini to parse and analyze the raw text
                processed_questions = process_text_with_gemini_for_quiz(raw_text)

                if processed_questions:
                    initial_count = len(st.session_state.questions)
                    added_count = 0
                    # Add new questions, preventing simple duplicates (based on text)
                    for new_q in processed_questions:
                        # Basic duplicate check - could be improved
                        if not any(item['question_text'].strip() == new_q['question_text'].strip() for item in st.session_state.questions):
                            st.session_state.questions.append(new_q)
                            added_count += 1
                        # else: # Too noisy
                            # st.warning(f"Skipping potential duplicate question found by Gemini: {new_q['question_text'][:50]}...")

                    if added_count > 0:
                        st.success(f"Successfully added {added_count} new questions (Total: {len(st.session_state.questions)}).")
                    elif len(st.session_state.questions) > initial_count:
                         st.info("No *new* unique questions were added from this input (potential duplicates skipped).")
                    else:
                         st.warning("Gemini processed text, but no new questions were added.")

                else:
                    st.warning("Gemini did not return any valid questions from the input text.")
            else:
                st.warning("No usable text was extracted from inputs.")


# --- Review Questions Page ---
elif page == "Review Questions":
    st.header("Review Added Questions")

    if not st.session_state.questions:
        st.info("No questions have been added yet. Use the 'Add Questions' page to add some.")
    else:
        st.write(f"Total questions added: {len(st.session_state.questions)}")
        st.markdown("---")

        for i, q in enumerate(st.session_state.questions):
            st.markdown(f"**Q{i+1}:** {q.get('question_text', 'Error: No question text found')}")

            options = q.get('options', {})
            correct_ans = q.get('correct_answer', 'N/A')
            explanations = q.get('explanations', {})

            if options:
                 st.markdown("**Options:**")
                 # Options should be sorted alphabetically by key from the parsing step
                 sorted_options = sorted(options.items())
                 for letter, text in sorted_options:
                      st.write(f"- {letter}) {text}")
            else:
                 st.warning("No options available for this question.")

            if correct_ans and correct_ans != 'N/A':
                 st.markdown(f"**Correct Answer:** {correct_ans}")
            else:
                 st.warning("Correct answer not determined by AI.")


            # Use an expander for explanations to keep the list of questions clean
            with st.expander(f"View Explanations for Q{i+1}"):
                 if explanations:
                     # Explanations should also be sorted by key
                     sorted_explanations = sorted(explanations.items())
                     for letter, explanation in sorted_explanations:
                          # Only show explanations for options present in the question's options
                          if letter in options: # Ensure the explanation key exists in the actual options
                             # Corrected the f-string here
                             st.markdown(f"- **{letter})**: {explanation}")
                 else:
                      st.info("No explanations available from AI.")

            st.markdown("---")


# --- Take Quiz Page ---
elif page == "Take Quiz":
    st.header("Take the Quiz")

    if not st.session_state.questions:
        st.warning("No questions available. Please add questions first on the 'Add Questions' page.")
    else:
        if st.session_state.quiz_state == 'not_started':
            st.write(f"Ready to take a quiz with {len(st.session_state.questions)} questions.")
            # Set default time limit in minutes
            st.session_state.time_limit = st.number_input("Set time limit (minutes):", min_value=1, value=st.session_state.time_limit, key="time_limit_input") # Added key
            if st.button("Start Quiz"):
                st.session_state.quiz_state = 'in_progress'
                st.session_state.user_answers = {} # Clear previous answers
                st.session_state.score = 0
                st.session_state.start_time = time.time()
                st.session_state.start_limit = st.session_state.time_limit * 60 # Store total seconds allowed
                st.rerun() # Corrected from experimental_rerun

        elif st.session_state.quiz_state == 'in_progress':
            current_time = time.time()
            elapsed_time = current_time - st.session_state.start_time
            remaining_time = st.session_state.start_limit - elapsed_time

            # Simple Timer Display (doesn't force submission, but disables submit when time is up)
            time_left = max(0, remaining_time)
            time_left_str = str(timedelta(seconds=int(time_left)))
            time_color = "red" if time_left <= 60 else "orange" if time_left <= 300 else "green"
            st.markdown(f"**<p style='color:{time_color};'>Time remaining: {time_left_str}</p>**", unsafe_allow_html=True)

            # Auto-submit logic is complex with Streamlit's rerun model.
            # Disabling the button is the simplest approach.
            submit_disabled = time_left <= 0 # Disable submit if time is 0 or less

            if time_left <= 0 and st.session_state.quiz_state == 'in_progress':
                 st.error("Time's up! Please submit your quiz.")
                 # Optional: Automatically move to completed state if timer hits 0 and quiz is in progress
                 # st.session_state.quiz_state = 'completed'
                 # st.session_state.end_time = time.time()
                 # st.rerun() # Rerun to show results

            st.subheader("Answer the Questions:")
            user_answers_current = st.session_state.user_answers.copy() # Use a copy for display update

            for i, q in enumerate(st.session_state.questions):
                st.markdown(f"**Q{i+1}:** {q.get('question_text', 'Error: No question text found')}")
                options = q.get('options', {}) # Use .get for safety

                if not options:
                     st.warning(f"Q{i+1} has no options parsed by Gemini. Skipping.")
                     user_answers_current[i] = "Skipped (No options)"
                     st.markdown("---")
                     continue

                try:
                     # Options are already sorted alphabetically by key from processing
                     sorted_options = sorted(options.items())
                     # Original option labels and keys
                     original_option_labels = [f"{letter}) {text}" for letter, text in sorted_options]
                     original_option_keys = [letter for letter, text in sorted_options]

                     # --- Add the placeholder option ---
                     placeholder_label = "--- Select an Option ---"
                     option_labels_with_placeholder = [placeholder_label] + original_option_labels
                     # No need to modify option_keys list, we map label back to original key

                except Exception as e:
                     st.error(f"Error processing options for Q{i+1}: {e}")
                     st.json(options)
                     user_answers_current[i] = "Error Processing Options"
                     st.markdown("---")
                     continue

                # Get current answer for this question, default to None
                current_answer = user_answers_current.get(i)

                # --- Determine the default index ---
                # Default to the placeholder index (0) if no answer is stored or it's invalid
                default_index = 0 # Start with the placeholder index
                # If the stored answer is one of the valid original keys
                if current_answer in original_option_keys:
                     try:
                         # Find the index of the current answer within the original keys list
                         # Add 1 because the placeholder is at index 0 in the display list
                         default_index = original_option_keys.index(current_answer) + 1
                     except ValueError:
                          # Should theoretically not happen if current_answer is in original_option_keys
                          default_index = 0 # Fallback to placeholder


                # Use st.radio with option labels including the placeholder
                selected_label = st.radio(
                    "Select your answer:",
                    options=option_labels_with_placeholder, # Use the list with placeholder
                    index=default_index,  # Use the calculated default index (0 or higher)
                    key=f"q_{i}_radio", # Unique key for each radio button
                    horizontal=False # Set to True if you prefer horizontal layout
                )

                # --- Extract and store the selected answer ---
                # Check if the selected label is the placeholder
                if selected_label == placeholder_label:
                     user_answers_current[i] = None # Store None if the placeholder is selected (no answer)
                else:
                     # Find the index of the selected label in the *original* option labels
                     # This safely gets the position *after* the placeholder was conceptually removed
                     try:
                         selected_index_in_original = original_option_labels.index(selected_label)
                         # Get the corresponding key (letter) from the original keys list
                         selected_letter = original_option_keys[selected_index_in_original]
                         user_answers_current[i] = selected_letter # Store the letter (A, B, C...)
                     except ValueError:
                         # This should ideally not happen if selected_label is not the placeholder
                         user_answers_current[i] = "Error Parsing Selection" # Indicate an unexpected selection


                st.markdown("---") # Separator between questions

            # Update session state *after* the loop to avoid issues during selection
            st.session_state.user_answers = user_answers_current

            # The submit button is disabled if time is up (handled above by submit_disabled var)
            if st.button("Submit Quiz", disabled=submit_disabled):
                st.session_state.quiz_state = 'completed'
                st.session_state.end_time = time.time() # Record submission time
                st.rerun() # Corrected from experimental_rerun

        elif st.session_state.quiz_state == 'completed':
            st.info("Quiz completed. Go to 'View Results' in the sidebar.")
            if st.button("Start New Quiz"):
                 reset_quiz()
                 st.rerun() # Corrected from experimental_rerun


# --- View Results Page ---
elif page == "View Results":
    st.header("Quiz Results")

    if st.session_state.quiz_state != 'completed':
        st.info("Please complete a quiz first on the 'Take Quiz' page.")
    elif not st.session_state.questions:
         st.warning("No questions available to review results for.")
    else:
        # Calculate score only when viewing results after completion
        correct_count = 0
        total_questions = len(st.session_state.questions)
        for i, q in enumerate(st.session_state.questions):
            user_ans = st.session_state.user_answers.get(i, "Not answered") # Default display string
            correct_ans = q.get('correct_answer', 'N/A') # Use .get() for safety

            # Only count if user answered and it matches the correct answer
            # Ensure user_ans is not None and not one of the error/skipped states
            if user_ans is not None and user_ans not in ["Not answered", "Skipped (No options)", "Error Processing Options", "Error Parsing Selection"] and \
               correct_ans is not None and user_ans.upper() == correct_ans.upper():
                correct_count += 1

        st.session_state.score = correct_count

        st.subheader(f"Your Score: {st.session_state.score}/{total_questions}")
        if st.session_state.start_time and st.session_state.end_time:
            quiz_duration = st.session_state.end_time - st.session_state.start_time
            st.write(f"Time taken: {str(timedelta(seconds=int(quiz_duration)))}")

        st.markdown("---")
        st.subheader("Review Answers and Explanations:")

        for i, q in enumerate(st.session_state.questions):
            user_ans = st.session_state.user_answers.get(i, "Not answered") # Default display string
            correct_ans = q.get('correct_answer', 'N/A')
            explanations = q.get('explanations', {})
            options = q.get('options', {})

            st.markdown(f"**Q{i+1}:** {q.get('question_text', 'Error: No question text available')}")

            # Display options with color indication
            if options:
                # Options should be sorted alphabetically by key from the parsing step
                sorted_options = sorted(options.items()) # Re-sort just in case
                for letter, text in sorted_options:
                    display_text = f"{letter}) {text}"
                    option_letter_upper = letter.upper()
                    correct_ans_upper = correct_ans.upper() if correct_ans != 'N/A' else 'NONE'
                    # Ensure user_ans is a valid string before calling upper()
                    user_ans_upper = user_ans.upper() if isinstance(user_ans, str) and user_ans not in ["Not answered", "Skipped (No options)", "Error Processing Options", "Error Parsing Selection"] else 'NONE'


                    if correct_ans_upper != 'NONE' and option_letter_upper == correct_ans_upper:
                        display_text = f"✅ **{display_text}** (Correct Answer)"
                    elif user_ans_upper != 'NONE' and option_letter_upper == user_ans_upper:
                         display_text = f"❌ **{display_text}** (Your Answer)"

                    st.write(display_text)
            else:
                 st.warning("No options available for this question.")


            # Display user's answer vs correct answer
            # Handle None and specific error strings for user_ans display
            if user_ans is None or user_ans == "Not answered":
                 st.markdown(f"**Your Answer:** Not answered")
            elif user_ans == "Skipped (No options)":
                 st.markdown(f"**Your Answer:** Skipped (No options parsed)")
            elif user_ans == "Error Processing Options":
                 st.markdown(f"**Your Answer:** Error Processing Options")
            elif user_ans == "Error Parsing Selection": # Handle this specific error from radio button
                 st.markdown(f"**Your Answer:** Error Parsing Selection")
            elif correct_ans == 'N/A':
                 st.markdown(f"**Your Answer:** {user_ans} - **Correct Answer Not Determined by AI**")
            elif user_ans.upper() == correct_ans.upper():
                 st.markdown(f"**Your Answer:** {user_ans} - **Correct**")
            else:
                 st.markdown(f"**Your Answer:** {user_ans} - **Incorrect** (Correct Answer: {correct_ans})")

            # Display Gemini explanations
            st.markdown("**Explanations:**")
            if explanations:
                sorted_explanations = sorted(explanations.items()) # Sort explanations by letter
                for letter, explanation in sorted_explanations:
                    # Only show explanations for options present in the question's options
                    if letter in options: # Ensure the explanation key exists in the actual options
                         # CORRECTED LINE HERE: Added {} around explanation
                         st.markdown(f"- **{letter})**: {explanation}")
                    else:
                        st.info("No explanations available from AI.")

            st.markdown("---")

        if st.button("Start New Quiz from Results Page"):
            reset_quiz()
            st.rerun() # Corrected from experimental_rerun