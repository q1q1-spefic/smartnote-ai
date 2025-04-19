# utils/flashcard_generator.py

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
import json
import csv
import os
import datetime
import math
import random
import tempfile
import shutil
import sqlite3
import zipfile
import base64
import io

class FlashcardGenerator:
    """Generate flashcards from documents for spaced repetition learning."""
    
    def __init__(self, llm=None, model_name="gpt-3.5-turbo", api_key=None):
        """Initialize the FlashcardGenerator with an LLM."""
        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=0.5
            )
        
        # Template for generating flashcards
        self.flashcard_template = PromptTemplate(
            input_variables=["text", "num_cards", "complexity"],
            template="""
            Create {num_cards} flashcards from the following text. Each flashcard should capture a key concept, fact, or relationship.
            Make the flashcards suitable for {complexity} complexity level.
            
            FORMAT YOUR RESPONSE AS A JSON ARRAY where each flashcard is a dictionary with the keys:
            - "front": The question or prompt side of the flashcard
            - "back": The answer side of the flashcard
            - "tags": Array of 1-3 tags/categories this flashcard belongs to
            
            TEXT:
            {text}
            
            FLASHCARDS:
            """
        )
        
        # Template for generating cloze deletion cards
        self.cloze_template = PromptTemplate(
            input_variables=["text", "num_cards", "complexity"],
            template="""
            Create {num_cards} cloze deletion flashcards from the following text.
            Each flashcard should be a complete sentence with a key term or phrase removed (indicated with [...]).
            Make the flashcards suitable for {complexity} complexity level.
            
            FORMAT YOUR RESPONSE AS A JSON ARRAY where each flashcard is a dictionary with the keys:
            - "text": The full text with the cloze deletion marked as [...]
            - "answer": The word or phrase that fills in the [...]
            - "tags": Array of 1-3 tags/categories this flashcard belongs to
            
            TEXT:
            {text}
            
            CLOZE CARDS:
            """
        )
        
        # Today's date for spaced repetition
        self.today = datetime.date.today()
        
        # Database for tracking reviews
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "data", "flashcards.sqlite")
        self._init_db()
    
    def _init_db(self):
        """Initialize the database for flashcard tracking."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            front TEXT,
            back TEXT,
            tags TEXT,
            deck_name TEXT,
            card_type TEXT,
            created_date TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id INTEGER,
            review_date TEXT,
            interval INTEGER,
            ease_factor REAL,
            next_review_date TEXT,
            FOREIGN KEY (card_id) REFERENCES flashcards (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _generate_cards(self, documents, card_type, num_cards=10, complexity="medium"):
        """Generate flashcards of specified type from documents."""
        # Concatenate document text with source info
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Limit text length to avoid token limits
        max_chars = 6000
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars]
        
        # Select appropriate template based on card type
        if card_type == "basic":
            template = self.flashcard_template
        elif card_type == "cloze":
            template = self.cloze_template
        else:
            raise ValueError(f"Unknown card type: {card_type}")
            
        # Create and run the chain
        chain = LLMChain(llm=self.llm, prompt=template)
        result = chain.run(text=combined_text, num_cards=num_cards, complexity=complexity)
        
        try:
            # Parse JSON result
            cards = json.loads(result)
            return cards
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            return {"error": "Failed to generate valid flashcards. Please try again."}
    
    def generate_flashcards(self, documents, config):
        """Generate a set of flashcards based on the configuration."""
        card_types = config.get("card_types", ["basic"])
        num_cards = config.get("num_cards", 10)
        complexity = config.get("complexity", "medium")
        deck_name = config.get("deck_name", "Knowledge Cards")
        
        all_cards = []
        
        # Distribute cards among selected types
        cards_per_type = max(1, num_cards // len(card_types))
        remaining = num_cards - (cards_per_type * len(card_types))
        
        for c_type in card_types:
            # Calculate how many cards of this type
            c_count = cards_per_type
            if remaining > 0:
                c_count += 1
                remaining -= 1
                
            # Generate cards
            type_cards = self._generate_cards(
                documents,
                c_type,
                num_cards=c_count,
                complexity=complexity
            )
            
            # Process cards based on type and add to database
            processed_cards = []
            for card in type_cards:
                if c_type == "basic":
                    processed_card = {
                        "front": card["front"],
                        "back": card["back"],
                        "tags": card["tags"],
                        "type": "basic"
                    }
                elif c_type == "cloze":
                    processed_card = {
                        "front": card["text"],
                        "back": card["answer"],
                        "tags": card["tags"],
                        "type": "cloze"
                    }
                
                # Add to database with initial spaced repetition values
                card_id = self._add_card_to_db(processed_card, deck_name)
                processed_card["id"] = card_id
                processed_cards.append(processed_card)
            
            all_cards.extend(processed_cards)
            
        return {
            "cards": all_cards,
            "deck_name": deck_name,
            "total_cards": len(all_cards)
        }
    
    def _add_card_to_db(self, card, deck_name):
        """Add a flashcard to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert card
        cursor.execute('''
        INSERT INTO flashcards (front, back, tags, deck_name, card_type, created_date)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            card["front"],
            card["back"],
            ','.join(card["tags"]),
            deck_name,
            card["type"],
            self.today.isoformat()
        ))
        
        card_id = cursor.lastrowid
        
        # Initialize review schedule (start tomorrow)
        next_review = self.today + datetime.timedelta(days=1)
        cursor.execute('''
        INSERT INTO reviews (card_id, review_date, interval, ease_factor, next_review_date)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            card_id,
            self.today.isoformat(),
            1,  # Initial interval of 1 day
            2.5,  # Initial ease factor
            next_review.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return card_id
    
    def get_cards_due_today(self):
        """Get all flashcards due for review today."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT f.id, f.front, f.back, f.tags, f.deck_name, f.card_type, r.interval, r.ease_factor
        FROM flashcards f
        JOIN reviews r ON f.id = r.card_id
        WHERE r.next_review_date <= ?
        ORDER BY r.next_review_date
        ''', (self.today.isoformat(),))
        
        cards = []
        for row in cursor.fetchall():
            card = {
                "id": row[0],
                "front": row[1],
                "back": row[2],
                "tags": row[3].split(','),
                "deck_name": row[4],
                "type": row[5],
                "interval": row[6],
                "ease_factor": row[7]
            }
            cards.append(card)
        
        conn.close()
        return cards
    
    def update_card_review(self, card_id, quality):
        """Update the spaced repetition schedule for a card after review.
        
        Quality rating:
        0 - Complete blackout (forgot completely)
        1 - Incorrect response but recognized answer
        2 - Incorrect response but answer seemed familiar
        3 - Correct but required significant effort
        4 - Correct after hesitation
        5 - Perfect recall
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current review data
        cursor.execute('''
        SELECT interval, ease_factor FROM reviews
        WHERE card_id = ?
        ORDER BY review_date DESC LIMIT 1
        ''', (card_id,))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False
            
        interval, ease_factor = result
        
        # SuperMemo SM-2 algorithm
        if quality < 3:
            # Reset interval to 1 if answer was wrong
            new_interval = 1
        else:
            # Adjust ease factor
            ease_factor += 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
            ease_factor = max(1.3, ease_factor)
            
            # Calculate next interval
            if interval == 1:
                new_interval = 6  # First success: 6 days
            elif interval == 6:
                new_interval = 15  # Second success: 15 days
            else:
                new_interval = round(interval * ease_factor)
                
        # Calculate next review date
        next_review = self.today + datetime.timedelta(days=new_interval)
        
        # Insert new review record
        cursor.execute('''
        INSERT INTO reviews (card_id, review_date, interval, ease_factor, next_review_date)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            card_id,
            self.today.isoformat(),
            new_interval,
            ease_factor,
            next_review.isoformat()
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def export_to_csv(self, deck_name=None):
        """Export flashcards to CSV format."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if deck_name:
            cursor.execute('''
            SELECT id, front, back, tags, card_type FROM flashcards
            WHERE deck_name = ?
            ''', (deck_name,))
        else:
            cursor.execute('''
            SELECT id, front, back, tags, card_type FROM flashcards
            ''')
            
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
            
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['front', 'back', 'tags', 'type'])
        
        # Write data
        for row in rows:
            writer.writerow([row[1], row[2], row[3], row[4]])
            
        return output.getvalue()
    
    def export_to_anki(self, deck_name=None):
        """Export flashcards to Anki-compatible format (.apkg)."""
        try:
            # We can only create a simplified template for Anki import
            # Full .apkg creation requires the genanki library which isn't standard
            
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Create collection.anki2 (SQLite database)
                db_path = os.path.join(temp_dir, "collection.anki2")
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create minimal tables required by Anki
                cursor.executescript('''
                CREATE TABLE col (
                    id integer PRIMARY KEY,
                    crt integer,
                    mod integer,
                    scm integer,
                    ver integer,
                    dty integer,
                    usn integer,
                    ls integer,
                    conf text,
                    models text,
                    decks text,
                    dconf text,
                    tags text
                );
                
                CREATE TABLE notes (
                    id integer PRIMARY KEY,
                    guid text,
                    mid integer,
                    mod integer,
                    usn integer,
                    tags text,
                    flds text,
                    sfld text,
                    csum integer,
                    flags integer,
                    data text
                );
                
                CREATE TABLE cards (
                    id integer PRIMARY KEY,
                    nid integer,
                    did integer,
                    ord integer,
                    mod integer,
                    usn integer,
                    type integer,
                    queue integer,
                    due integer,
                    ivl integer,
                    factor integer,
                    reps integer,
                    lapses integer,
                    left integer,
                    odue integer,
                    odid integer,
                    flags integer,
                    data text
                );
                ''')
                
                conn.commit()
                conn.close()
                
                # Create a zip file (Anki packages are just zip files)
                output_path = os.path.join(temp_dir, f"{deck_name or 'flashcards'}.apkg")
                with zipfile.ZipFile(output_path, 'w') as zipf:
                    zipf.write(db_path, arcname="collection.anki2")
                    
                    # Create a media file to indicate empty media
                    media_path = os.path.join(temp_dir, "media")
                    with open(media_path, 'w') as f:
                        f.write("{}")
                    zipf.write(media_path, arcname="media")
                
                # Read the zip file into memory
                with open(output_path, 'rb') as f:
                    apkg_data = f.read()
                
                # Encode as base64 for transfer
                return base64.b64encode(apkg_data).decode('utf-8')
                
            finally:
                # Clean up temp directory
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            return f"Error creating Anki package: {str(e)}"
