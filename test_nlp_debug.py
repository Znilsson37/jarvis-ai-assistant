"""Debug script for NLP module"""

from modules.nlp import NLPProcessor

def test_nlp_debug():
    nlp = NLPProcessor()
    
    # Test intent recognition
    print("=== Intent Recognition Tests ===")
    test_cases = [
        "open chrome browser",
        "what's the system status", 
        "search for latest news",
        "check my email"
    ]
    
    for text in test_cases:
        result = nlp.recognize_intent(text)
        print(f"Text: '{text}'")
        print(f"Result: {result}")
        print()
    
    # Test sentiment analysis
    print("=== Sentiment Analysis Tests ===")
    sentiment_tests = [
        "This is great and wonderful!",
        "This is terrible and horrible.",
        "The weather is cloudy today."
    ]
    
    for text in sentiment_tests:
        sentiment = nlp.analyze_sentiment(text)
        print(f"Text: '{text}'")
        print(f"Sentiment: {sentiment}")
        print()
    
    # Test task planning
    print("=== Task Planning Tests ===")
    planning_tests = [
        "open chrome and search for weather",
        "check system status"
    ]
    
    for command in planning_tests:
        result = nlp.plan_task(command)
        print(f"Command: '{command}'")
        print(f"Result: {result}")
        print()

if __name__ == "__main__":
    test_nlp_debug()
