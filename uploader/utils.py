
from transformers import pipeline

def criticValidatorEvaluator(llm_pipe, question, answer, promptTemplate, tokenizer, retriever):
    # Retrieve context
    retrieved_docs = retriever.invoke(question)
    print(retrieved_docs)
    context = "\n".join([d.page_content for d in retrieved_docs])
    
    # Fill prompt
    prompt = promptTemplate.format(
        question=question,
        context=context,
        answer=answer
    )
    output = llm_pipe(prompt, eos_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    return output, pipeline


def sorcerer(llm_pipe, accuracies, inaccuracies, promptTemplate, model, tokenizer, retriever, temperature = 0.5,
            top_p = 0.5, max_tokens = 200, seed = 42):

    # Fill prompt
    prompt = promptTemplate.format(
        accuracies=accuracies,
        inaccuracies=inaccuracies
    )
    # print (prompt)
    output = llm_pipe(prompt, eos_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    return output, pipeline