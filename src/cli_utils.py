"""CLI utility functions for user interaction."""


def get_user_input_with_confirmation(prompt: str = "Your response (or 'quit' to exit): ") -> str:
    """
    Get user input with echo-back confirmation for longer responses.
    
    This helps catch terminal readline buffer corruption that can occur in some
    IDE terminals when users edit their input with arrow keys or backspace.
    The terminal may display one thing but Python's input() receives corrupted data.
    """
    user_input = input(prompt)
    
    # For longer responses (where corruption is more likely), echo back and offer re-entry
    if len(user_input) > 30 and user_input.lower() != 'quit':
        print(f"\n📝 Captured ({len(user_input)} chars):")
        print(user_input)
        print()
        confirm = input("Press Enter to confirm, or 'r' to re-enter: ").strip().lower()
        if confirm == 'r':
            user_input = input("Re-enter your response: ")
            # Show the re-entered response too for verification
            if len(user_input) > 30:
                print(f"\n📝 Captured ({len(user_input)} chars):")
                print(user_input)
    
    return user_input
