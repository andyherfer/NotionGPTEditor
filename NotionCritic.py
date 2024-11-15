import os
import hashlib
import json
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import re

import openai
from notion_client import Client
from dotenv import load_dotenv
from openai import OpenAI


class NotionCritic:
    def __init__(self, database_id: str, check_interval: int = 300):
        # Load environment variables
        load_dotenv()

        # Initialize APIs
        self.notion = Client(auth=os.getenv("NOTION_TOKEN"))
        self.database_id = database_id
        self.check_interval = check_interval  # Time between checks in seconds
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )  # Create OpenAI client instance

        # Setup logging
        self.setup_logging()

        # Initialize SQLite database with datetime adapter
        sqlite3.register_adapter(datetime, lambda x: x.isoformat())
        sqlite3.register_converter(
            "datetime", lambda x: datetime.fromisoformat(x.decode())
        )
        self.init_database()

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "notion_critic.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def init_database(self):
        """Initialize SQLite database with necessary tables."""
        conn = sqlite3.connect("notion_tracker.db")
        cursor = conn.cursor()

        # Create table for page versions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS page_versions (
                page_id TEXT,
                version_hash TEXT,
                content TEXT,
                timestamp DATETIME,
                PRIMARY KEY (page_id, version_hash)
            )
        """
        )

        # Create table for change suggestions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS change_suggestions (
                page_id TEXT,
                old_hash TEXT,
                new_hash TEXT,
                suggestion TEXT,
                timestamp DATETIME,
                PRIMARY KEY (page_id, old_hash, new_hash)
            )
        """
        )

        conn.commit()
        conn.close()

    def get_page_content(self, page_id: str) -> dict:
        """Fetch page content from Notion."""
        return self.notion.pages.retrieve(page_id)

    def compute_hash(self, content: dict) -> str:
        """Compute hash of page content."""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def get_latest_version(self, page_id: str) -> Optional[Tuple[str, str]]:
        """Get the latest version hash and content for a page."""
        conn = sqlite3.connect("notion_tracker.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT version_hash, content
            FROM page_versions
            WHERE page_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """,
            (page_id,),
        )

        result = cursor.fetchone()
        conn.close()

        return result if result else None

    def store_version(self, page_id: str, version_hash: str, content: dict):
        """Store a new version of a page."""
        conn = sqlite3.connect("notion_tracker.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO page_versions (page_id, version_hash, content, timestamp)
            VALUES (?, ?, ?, ?)
        """,
            (page_id, version_hash, json.dumps(content), datetime.now()),
        )

        conn.commit()
        conn.close()

    def get_changes(self, old_content: dict, new_content: dict) -> str:
        """Compare old and new content to generate a description of changes."""
        changes = []

        # Extract and compare title
        old_title = (
            old_content.get("properties", {})
            .get("Name", {})
            .get("title", [{}])[0]
            .get("text", {})
            .get("content", "")
        )
        new_title = (
            new_content.get("properties", {})
            .get("Name", {})
            .get("title", [{}])[0]
            .get("text", {})
            .get("content", "")
        )

        if old_title != new_title:
            changes.append(f"Title changed from '{old_title}' to '{new_title}'")

        # Extract and compare properties
        old_props = old_content.get("properties", {})
        new_props = new_content.get("properties", {})

        # Compare status
        old_status = old_props.get("Status", {}).get("select", {}).get("name", "")
        new_status = new_props.get("Status", {}).get("select", {}).get("name", "")
        if old_status != new_status:
            changes.append(f"Status changed from '{old_status}' to '{new_status}'")

        # Compare content blocks
        try:
            old_blocks = self.notion.blocks.children.list(block_id=old_content["id"])
            new_blocks = self.notion.blocks.children.list(block_id=new_content["id"])

            old_text = self._extract_text_from_blocks(old_blocks.get("results", []))
            new_text = self._extract_text_from_blocks(new_blocks.get("results", []))

            if old_text != new_text:
                changes.append("Content changes:")
                changes.append("Old content:")
                changes.append(old_text)
                changes.append("\nNew content:")
                changes.append(new_text)
        except Exception as e:
            self.logger.error(f"Error comparing blocks: {str(e)}")
            changes.append("Unable to compare page content blocks")

        if not changes:
            return "No significant changes detected in the page content."

        return "\n".join(changes)

    def _extract_text_from_blocks(self, blocks: list) -> str:
        """Extract text content from Notion blocks."""
        text_content = []

        for block in blocks:
            block_type = block.get("type", "")
            if block_type == "paragraph":
                text = self._get_rich_text_content(
                    block.get("paragraph", {}).get("rich_text", [])
                )
                if text:
                    text_content.append(text)
            elif block_type == "heading_1":
                text = self._get_rich_text_content(
                    block.get("heading_1", {}).get("rich_text", [])
                )
                if text:
                    text_content.append(f"# {text}")
            elif block_type == "heading_2":
                text = self._get_rich_text_content(
                    block.get("heading_2", {}).get("rich_text", [])
                )
                if text:
                    text_content.append(f"## {text}")
            elif block_type == "heading_3":
                text = self._get_rich_text_content(
                    block.get("heading_3", {}).get("rich_text", [])
                )
                if text:
                    text_content.append(f"### {text}")
            elif block_type == "bulleted_list_item":
                text = self._get_rich_text_content(
                    block.get("bulleted_list_item", {}).get("rich_text", [])
                )
                if text:
                    text_content.append(f"• {text}")
            elif block_type == "numbered_list_item":
                text = self._get_rich_text_content(
                    block.get("numbered_list_item", {}).get("rich_text", [])
                )
                if text:
                    text_content.append(f"1. {text}")

        return "\n".join(text_content)

    def _get_rich_text_content(self, rich_text: list) -> str:
        """Extract text content from rich text array."""
        return " ".join(
            text.get("text", {}).get("content", "")
            for text in rich_text
            if text.get("type") == "text"
        )

    def get_gpt4_suggestions(self, changes: str, current_content: dict) -> str:
        """Get brief, actionable suggestions from GPT-4."""
        try:
            blocks = self.notion.blocks.children.list(block_id=current_content["id"])
            full_content = self._extract_text_from_blocks(blocks.get("results", []))

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a direct editor. Provide only essential feedback as bullet points.
                        • Focus on critical improvements
                        • Keep suggestions brief and actionable
                        • Maximum 3-4 bullet points
                        • No explanations - just changes needed""",
                    },
                    {
                        "role": "user",
                        "content": f"Review and suggest critical improvements:\n{full_content}",
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error getting GPT-4 suggestions: {str(e)}")
            return "Error generating suggestions."

    def implement_suggestions(
        self, page_id: str, suggestions: str, current_content: dict
    ):
        """Implement suggestions concisely."""
        try:
            blocks = self.notion.blocks.children.list(block_id=page_id)
            current_text = self._extract_text_from_blocks(blocks.get("results", []))

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a direct editor. Rewrite content:
                        • Implement suggested changes
                        • Keep it brief and clear
                        • Use bullet points where possible
                        • No explanations or commentary""",
                    },
                    {
                        "role": "user",
                        "content": f"Content: {current_text}\nSuggestions: {suggestions}\nRewrite implementing changes:",
                    },
                ],
            )

            new_content = response.choices[0].message.content
            new_blocks = self._convert_markdown_to_blocks(new_content)

            # Update page
            existing_blocks = self.notion.blocks.children.list(block_id=page_id)
            for block in existing_blocks.get("results", []):
                self.notion.blocks.delete(block_id=block["id"])
            self.notion.blocks.children.append(block_id=page_id, children=new_blocks)

        except Exception as e:
            self.logger.error(f"Failed to implement suggestions: {str(e)}")
            raise

    def _convert_markdown_to_blocks(self, markdown_content: str) -> list:
        """Convert markdown content to Notion blocks."""
        blocks = []
        lines = markdown_content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# "):
                blocks.append(
                    {
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[2:]}}
                            ]
                        },
                    }
                )
            elif line.startswith("## "):
                blocks.append(
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[3:]}}
                            ]
                        },
                    }
                )
            elif line.startswith("### "):
                blocks.append(
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[4:]}}
                            ]
                        },
                    }
                )
            elif line.startswith("- "):
                blocks.append(
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[2:]}}
                            ]
                        },
                    }
                )
            elif re.match(r"^\d+\. ", line):
                content = line[line.index(".") + 2 :]
                blocks.append(
                    {
                        "object": "block",
                        "type": "numbered_list_item",
                        "numbered_list_item": {
                            "rich_text": [
                                {"type": "text", "text": {"content": content}}
                            ]
                        },
                    }
                )
            else:
                blocks.append(
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": line}}]
                        },
                    }
                )

        return blocks

    def store_suggestion(
        self, page_id: str, old_hash: str, new_hash: str, suggestion: str
    ):
        """Store a suggestion in the database and update the Notion page."""
        # Store in SQLite
        conn = sqlite3.connect("notion_tracker.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO change_suggestions (page_id, old_hash, new_hash, suggestion, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (page_id, old_hash, new_hash, suggestion, datetime.now()),
        )

        conn.commit()
        conn.close()

        try:
            self.logger.info(f"Processing suggestions for page {page_id}")

            # First add the suggestions as comments
            self._add_suggestion_comments(page_id, suggestion)

            # Then implement the suggestions
            current_content = self.get_page_content(page_id)
            self.implement_suggestions(page_id, suggestion, current_content)

        except Exception as e:
            self.logger.error(
                f"Error processing suggestions for page {page_id}: {str(e)}"
            )

    def _add_suggestion_comments(self, page_id: str, suggestion: str):
        """Add suggestions as comments to the page."""
        blocks = []

        # Add header for suggestions
        blocks.append(
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"💡 AI Suggestions ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
                            },
                        }
                    ],
                    "icon": {"type": "emoji", "emoji": "💡"},
                    "color": "blue_background",
                },
            }
        )

        # Add the suggestions
        blocks.append(
            {
                "object": "block",
                "type": "quote",
                "quote": {
                    "rich_text": [{"type": "text", "text": {"content": suggestion}}],
                    "color": "default",
                },
            }
        )

        # Add divider
        blocks.append({"object": "block", "type": "divider", "divider": {}})

        # Append blocks to the page
        self.notion.blocks.children.append(block_id=page_id, children=blocks)

    def process_database(self):
        """Process all pages in the Notion database."""
        try:
            pages = self.notion.databases.query(database_id=self.database_id)

            for page in pages["results"]:
                try:
                    page_id = page["id"]
                    self.logger.info(f"Processing page: {page_id}")

                    current_content = self.get_page_content(page_id)
                    current_hash = self.compute_hash(current_content)

                    # Get the latest version from our database
                    latest_version = self.get_latest_version(page_id)

                    if not latest_version or latest_version[0] != current_hash:
                        # We found a change or new page
                        if latest_version:
                            # This is a change to an existing page
                            self.logger.info(f"Changes detected in page: {page_id}")
                            old_content = json.loads(latest_version[1])
                            changes = self.get_changes(old_content, current_content)
                            suggestions = self.get_gpt4_suggestions(
                                changes, current_content
                            )

                            # Store and update Notion page with suggestions
                            self.store_suggestion(
                                page_id, latest_version[0], current_hash, suggestions
                            )
                        else:
                            self.logger.info(f"New page detected: {page_id}")
                            # For new pages, still provide initial suggestions
                            suggestions = self.get_gpt4_suggestions(
                                "New page", current_content
                            )
                            self.store_suggestion(
                                page_id, "", current_hash, suggestions
                            )

                        # Store the new version
                        self.store_version(page_id, current_hash, current_content)

                except Exception as e:
                    self.logger.error(f"Error processing page {page_id}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Error querying database: {str(e)}")

    def monitor(self):
        """Continuously monitor the Notion database for changes."""
        self.logger.info(f"Starting monitoring of database: {self.database_id}")
        self.logger.info(f"Check interval: {self.check_interval} seconds")

        while True:
            try:
                self.process_database()
                self.logger.info("Waiting for next check...")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break

            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                self.logger.info("Retrying in 60 seconds...")
                time.sleep(60)


def main():
    # Load environment variables
    load_dotenv()

    # Initialize NotionCritic with your database ID
    database_id = os.getenv("NOTION_DATABASE_ID")
    check_interval = int(os.getenv("CHECK_INTERVAL", "300"))  # Default to 5 minutes

    critic = NotionCritic(database_id, check_interval)

    # Start monitoring
    critic.monitor()


if __name__ == "__main__":
    main()
