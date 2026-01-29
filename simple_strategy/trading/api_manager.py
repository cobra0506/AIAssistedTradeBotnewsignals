import json
import os
import hashlib
from pathlib import Path

class APIManager:
    def __init__(self):
        # Set the path to the API accounts file
        self.accounts_file = os.path.join(os.path.dirname(__file__), 'api_accounts.json')
        # Create the file if it doesn't exist
        self._ensure_accounts_file_exists()
    
    def _ensure_accounts_file_exists(self):
        """Create the accounts file if it doesn't exist"""
        if not os.path.exists(self.accounts_file):
            # Create empty structure
            empty_accounts = {
                "demo_accounts": {},
                "live_accounts": {}
            }
            self._save_accounts(empty_accounts)
    
    def _save_accounts(self, accounts):
        """Save accounts to file"""
        with open(self.accounts_file, 'w') as f:
            json.dump(accounts, f, indent=4)
    
    def _load_accounts(self):
        """Load accounts from file"""
        with open(self.accounts_file, 'r') as f:
            return json.load(f)
    
    def get_all_accounts(self):
        """Get all accounts (both demo and live)"""
        return self._load_accounts()
    
    def add_demo_account(self, name, api_key, api_secret, description=""):
        """Add a new demo account"""
        accounts = self._load_accounts()
        accounts["demo_accounts"][name] = {
            "api_key": api_key,
            "api_secret": api_secret,
            "description": description,
            "testnet": True
        }
        self._save_accounts(accounts)
        return True
    
    def add_live_account(self, name, api_key, api_secret, description=""):
        """Add a new live account"""
        accounts = self._load_accounts()
        accounts["live_accounts"][name] = {
            "api_key": api_key,
            "api_secret": api_secret,
            "description": description,
            "testnet": False
        }
        self._save_accounts(accounts)
        return True
    
    def update_demo_account(self, name, api_key, api_secret, description=""):
        """Update an existing demo account"""
        accounts = self._load_accounts()
        if name in accounts["demo_accounts"]:
            accounts["demo_accounts"][name] = {
                "api_key": api_key,
                "api_secret": api_secret,
                "description": description,
                "testnet": True
            }
            self._save_accounts(accounts)
            return True
        return False
    
    def update_live_account(self, name, api_key, api_secret, description=""):
        """Update an existing live account"""
        accounts = self._load_accounts()
        if name in accounts["live_accounts"]:
            accounts["live_accounts"][name] = {
                "api_key": api_key,
                "api_secret": api_secret,
                "description": description,
                "testnet": False
            }
            self._save_accounts(accounts)
            return True
        return False
    
    def delete_demo_account(self, name):
        """Delete a demo account"""
        accounts = self._load_accounts()
        if name in accounts["demo_accounts"]:
            del accounts["demo_accounts"][name]
            self._save_accounts(accounts)
            return True
        return False
    
    def delete_live_account(self, name):
        """Delete a live account"""
        accounts = self._load_accounts()
        if name in accounts["live_accounts"]:
            del accounts["live_accounts"][name]
            self._save_accounts(accounts)
            return True
        return False
    
    def get_demo_account_names(self):
        """Get all demo account names"""
        accounts = self._load_accounts()
        return list(accounts["demo_accounts"].keys())
    
    def get_live_account_names(self):
        """Get all live account names"""
        accounts = self._load_accounts()
        return list(accounts["live_accounts"].keys())
    
    def get_demo_account(self, name):
        """Get a specific demo account"""
        accounts = self._load_accounts()
        return accounts["demo_accounts"].get(name, None)
    
    def get_live_account(self, name):
        """Get a specific live account"""
        accounts = self._load_accounts()
        return accounts["live_accounts"].get(name, None)