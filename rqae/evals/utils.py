def display_messages(system_prompt, messages):
    result = ""
    result += "=" * 20 + "[[ SYSTEM PROMPT ]]" + "=" * 20 + "\n"
    result += system_prompt + "\n"
    for message in messages:
        result += "=" * 20 + "[[ " + message["role"].upper() + " ]]" + "=" * 20 + "\n"
        result += message["content"][0]["text"] + "\n"
    return result
