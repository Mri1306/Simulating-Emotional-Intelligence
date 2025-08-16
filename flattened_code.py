import json
import sys
import logging
from typing import List, Dict, Any, Optional


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metadata_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def flatten_metadata(input_data: List[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    Flatten the nested metadata structure into a simplified format with input and output pairs.
    
    Args:
        input_data (list): The nested metadata structure with an outer list for each person
        
    Returns:
        list: Flattened data in the desired format
    """
    flattened_data = []
    skipped_entries = 0
    processed_entries = 0
    
    try:
      
        total_people = len(input_data)
        
        for person_idx, person_data in enumerate(input_data):
            logging.info(f"Processing person {person_idx + 1}/{total_people}")
            
            
            if not isinstance(person_data, list):
                logging.warning(f"Skipping invalid person format at index {person_idx}")
                skipped_entries += 1
                continue
            
            person_id = f"Person_{person_idx + 1}"
            
            for group_idx, group in enumerate(person_data):
               
                if not isinstance(group, dict) or "scenario" not in group or "questions" not in group:
                    logging.warning(f"Skipping invalid group format for {person_id}, group index {group_idx}")
                    skipped_entries += 1
                    continue
                    
                scenario = group.get("scenario", "Unknown Group")
                
                
                questions = group.get("questions", [])
                if not isinstance(questions, list):
                    logging.warning(f"Invalid questions format in {person_id}, scenario '{scenario}', skipping")
                    skipped_entries += 1
                    continue
                
                for question_idx, question_data in enumerate(questions):
                    try:
                        
                        question_text = None
                        if "main_question" in question_data:
                            question_text = question_data["main_question"]
                        elif "follow_up_question" in question_data:
                            question_text = question_data["follow_up_question"]
                        
                        if not question_text:
                            logging.warning(f"No question found in {person_id}, scenario '{scenario}', question index {question_idx}")
                            skipped_entries += 1
                            continue
                        
                        
                        answer_text = question_data.get("answer", "")
                        if not answer_text:
                            logging.warning(f"Missing answer for question: '{question_text}' in {person_id}, scenario '{scenario}'")
                            
                        
                        
                        metadata = question_data.get("metadata", {})
                        if not metadata:
                            logging.warning(f"No metadata found for question: '{question_text}' in {person_id}, scenario '{scenario}'")
                        
                        
                        input_string = f"Q: {question_text}\n"
                        
                        
                        speech_attr = metadata.get("speech_attributes", "")
                        if speech_attr:
                            input_string += f"Tone: {speech_attr.lower()}\n"
                        
                        response_time = metadata.get("response_time", "")
                        if response_time:
                            input_string += f"Response Time: {response_time}\n"
                        
                        body_language = metadata.get("body_language", "")
                        if body_language:
                            input_string += f"Body Language: {body_language.lower()}\n"
                        
                        input_string += "A:"
                        
                        
                        flattened_entry = {
                            "input": input_string,
                            "output": answer_text,
                            
                            "person_id": person_id
                        }
                        
                        flattened_data.append(flattened_entry)
                        processed_entries += 1
                        
                    except Exception as e:
                        skipped_entries += 1
                        logging.error(f"Error processing question in {person_id}, scenario '{scenario}': {str(e)}")
                        continue
                    
    except Exception as e:
        logging.error(f"Unexpected error during data flattening: {str(e)}")
    
    logging.info(f"Processing complete. Processed {processed_entries} entries, skipped {skipped_entries} entries.")
    
    
    for entry in flattened_data:
        if "person_id" in entry:
            del entry["person_id"]
            
    return flattened_data

def main():
    try:
        
        input_file = 'meta_data.json'
        output_file = 'flattened_metadata.json'
        
        logging.info(f"Reading data from {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON format in {input_file}: {str(e)}")
                    sys.exit(1)
        except FileNotFoundError:
            logging.error(f"Input file {input_file} not found")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error reading input file: {str(e)}")
            sys.exit(1)
        
        
        if not isinstance(metadata, list):
            logging.error("Input data must be a list of person data")
            sys.exit(1)
        
        logging.info(f"Starting to flatten data for {len(metadata)} people")
        
        
        flattened_data = flatten_metadata(metadata)
        
        if not flattened_data:
            logging.warning("No data was successfully flattened")
            
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(flattened_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully wrote {len(flattened_data)} flattened entries to {output_file}")
        except Exception as e:
            logging.error(f"Error writing output file: {str(e)}")
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
