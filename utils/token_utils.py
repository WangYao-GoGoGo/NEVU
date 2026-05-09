import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens




if __name__ == '__main__':
    test = "**[Key Concepts:]** Human Value: Human Values are fundamental principles or ideals that guide decision-making, behavior, and interactions, reflecting what individuals and societies consider deeply important. These values transcend specific situations, serving as a framework for evaluating actions, people, and events. They are organized hierarchically based on their relative importance and are closely tied to motivational goals, such as self-enhancement, self-transcendence, openness to change, and conservation. Encoded as mental representations, human values are imbued with context-sensitive affect, influencing attitudes, societal norms, moral reasoning, and personal well-being. While often universal, they vary in expression across cultures and contexts, providing a foundation for understanding human needs, goals, and relationships. Event: An event is an action, series of actions, or a change occurring at a specific time due to certain reasons, involving associated entities such as people, objects, and locations. Events are described using the 5W1H dimensions (Who, What, Where, When, Why, and How) and can vary in complexity, encompassing both standalone occurrences and interconnected sequences that form a broader narrative. Subevent: A subevent is a component or subordinate part of a larger event, providing detailed and finer-grained information about specific actions, changes, or outcomes within the context of the main event. Subevents exist in hierarchical relationships with their parent events and contribute to a comprehensive understanding and analysis of the overarching event."
    token_count = num_tokens_from_string(test, "cl100k_base")
    print(token_count)
