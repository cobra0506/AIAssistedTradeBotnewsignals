import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simple_strategy.trading.api_manager import APIManager

class APIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("API Account Manager")
        self.root.geometry("600x500")
        
        # Create API Manager instance
        self.manager = APIManager()
        
        # Create main GUI
        self.create_widgets()
        
        # Refresh account lists
        self.refresh_account_lists()
    
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔑 API Account Management", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Demo Accounts Tab
        demo_frame = ttk.Frame(notebook)
        notebook.add(demo_frame, text="Demo Accounts")
        self.create_account_tab(demo_frame, "demo")
        
        # Live Accounts Tab
        live_frame = ttk.Frame(notebook)
        notebook.add(live_frame, text="Live Accounts")
        self.create_account_tab(live_frame, "live")
        
        # Bottom buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(bottom_frame, text="Close", 
                  command=self.root.quit).pack(side=tk.RIGHT)
    
    def create_account_tab(self, parent, account_type):
        # Frame for account list
        list_frame = ttk.LabelFrame(parent, text=f"{account_type.title()} Accounts", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create treeview for account list
        columns = ('Name', 'Description')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        # Define headings
        tree.heading('Name', text='Account Name')
        tree.heading('Description', text='Description')
        
        # Configure column widths
        tree.column('Name', width=150)
        tree.column('Description', width=300)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store tree reference
        if account_type == "demo":
            self.demo_tree = tree
        else:
            self.live_tree = tree
        
        # Buttons frame
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X)
        
        # Add buttons
        ttk.Button(button_frame, text="Add Account", 
                  command=lambda: self.add_account(account_type)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Edit Account", 
                  command=lambda: self.edit_account(account_type)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Account", 
                  command=lambda: self.delete_account(account_type)).pack(side=tk.LEFT, padx=5)
    
    def refresh_account_lists(self):
        # Clear existing items
        for item in self.demo_tree.get_children():
            self.demo_tree.delete(item)
        for item in self.live_tree.get_children():
            self.live_tree.delete(item)
        
        # Add demo accounts
        demo_names = self.manager.get_demo_account_names()
        for name in demo_names:
            account = self.manager.get_demo_account(name)
            self.demo_tree.insert('', tk.END, values=(name, account['description']))
        
        # Add live accounts
        live_names = self.manager.get_live_account_names()
        for name in live_names:
            account = self.manager.get_live_account(name)
            self.live_tree.insert('', tk.END, values=(name, account['description']))
    
    def add_account(self, account_type):
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Add {account_type.title()} Account")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create form
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Account Name
        ttk.Label(frame, text="Account Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_entry = ttk.Entry(frame, width=30)
        name_entry.grid(row=0, column=1, pady=5)
        
        # API Key
        ttk.Label(frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, pady=5)
        key_entry = ttk.Entry(frame, width=30, show="*")
        key_entry.grid(row=1, column=1, pady=5)
        
        # API Secret
        ttk.Label(frame, text="API Secret:").grid(row=2, column=0, sticky=tk.W, pady=5)
        secret_entry = ttk.Entry(frame, width=30, show="*")
        secret_entry.grid(row=2, column=1, pady=5)
        
        # Description
        ttk.Label(frame, text="Description:").grid(row=3, column=0, sticky=tk.W, pady=5)
        desc_entry = ttk.Entry(frame, width=30)
        desc_entry.grid(row=3, column=1, pady=5)
        
        def save_account():
            name = name_entry.get().strip()
            api_key = key_entry.get().strip()
            api_secret = secret_entry.get().strip()
            description = desc_entry.get().strip()
            
            if not name or not api_key or not api_secret:
                messagebox.showerror("Error", "Please fill in all required fields (Name, API Key, API Secret)")
                return
            
            try:
                if account_type == "demo":
                    self.manager.add_demo_account(name, api_key, api_secret, description)
                else:
                    self.manager.add_live_account(name, api_key, api_secret, description)
                
                self.refresh_account_lists()
                dialog.destroy()
                messagebox.showinfo("Success", f"{account_type.title()} account added successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add account: {str(e)}")
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Save", command=save_account).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def edit_account(self, account_type):
        # Get selected account
        tree = self.demo_tree if account_type == "demo" else self.live_tree
        selection = tree.selection()
        
        if not selection:
            messagebox.showwarning("Warning", "Please select an account to edit")
            return
        
        # Get account name and details
        item = tree.item(selection[0])
        account_name = item['values'][0]
        
        if account_type == "demo":
            account = self.manager.get_demo_account(account_name)
        else:
            account = self.manager.get_live_account(account_name)
        
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {account_type.title()} Account")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create form
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Account Name (read-only)
        ttk.Label(frame, text="Account Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_label = ttk.Label(frame, text=account_name, font=("Arial", 10, "bold"))
        name_label.grid(row=0, column=1, pady=5, sticky=tk.W)
        
        # API Key
        ttk.Label(frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, pady=5)
        key_entry = ttk.Entry(frame, width=30, show="*")
        key_entry.grid(row=1, column=1, pady=5)
        key_entry.insert(0, account['api_key'])
        
        # API Secret
        ttk.Label(frame, text="API Secret:").grid(row=2, column=0, sticky=tk.W, pady=5)
        secret_entry = ttk.Entry(frame, width=30, show="*")
        secret_entry.grid(row=2, column=1, pady=5)
        secret_entry.insert(0, account['api_secret'])
        
        # Description
        ttk.Label(frame, text="Description:").grid(row=3, column=0, sticky=tk.W, pady=5)
        desc_entry = ttk.Entry(frame, width=30)
        desc_entry.grid(row=3, column=1, pady=5)
        desc_entry.insert(0, account['description'])
        
        def update_account():
            api_key = key_entry.get().strip()
            api_secret = secret_entry.get().strip()
            description = desc_entry.get().strip()
            
            if not api_key or not api_secret:
                messagebox.showerror("Error", "Please fill in all required fields (API Key, API Secret)")
                return
            
            try:
                if account_type == "demo":
                    self.manager.update_demo_account(account_name, api_key, api_secret, description)
                else:
                    self.manager.update_live_account(account_name, api_key, api_secret, description)
                
                self.refresh_account_lists()
                dialog.destroy()
                messagebox.showinfo("Success", f"{account_type.title()} account updated successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update account: {str(e)}")
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Update", command=update_account).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def delete_account(self, account_type):
        # Get selected account
        tree = self.demo_tree if account_type == "demo" else self.live_tree
        selection = tree.selection()
        
        if not selection:
            messagebox.showwarning("Warning", "Please select an account to delete")
            return
        
        # Get account name
        item = tree.item(selection[0])
        account_name = item['values'][0]
        
        # Confirm deletion
        result = messagebox.askyesno("Confirm Delete", 
                                     f"Are you sure you want to delete '{account_name}'?")
        
        if result:
            try:
                if account_type == "demo":
                    self.manager.delete_demo_account(account_name)
                else:
                    self.manager.delete_live_account(account_name)
                
                self.refresh_account_lists()
                messagebox.showinfo("Success", f"{account_type.title()} account deleted successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete account: {str(e)}")

# Test function
def test_api_gui():
    root = tk.Tk()
    app = APIGUI(root)
    root.mainloop()

if __name__ == "__main__":
    test_api_gui()