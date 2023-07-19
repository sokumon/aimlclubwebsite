import { OpenAI } from 'langchain/llms'
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains'
import { HNSWLib } from 'langchain/vectorstores'
import { PromptTemplate } from 'langchain/prompts'

const CONDENSE_PROMPT =
    PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`)

const QA_PROMPT = PromptTemplate.fromTemplate(
    `You are a helpful and smart AI Assistant for the APSIT AIML Club. You will answer with the help of the provided piece of information. You will summarize the information and answer the question in a maximum of 3 sentences.
Question: {question}
=========
{context}
=========
Answer in Markdown:`
)

export const makeChain = (
    vectorstore: HNSWLib,
    onTokenStream?: (token: string) => void
) => {
    const questionGenerator = new LLMChain({
        llm: new OpenAI({ temperature: 0 }),
        prompt: CONDENSE_PROMPT,
    })
    const docChain = loadQAChain(
        new OpenAI({
            temperature: 0,
            streaming: Boolean(onTokenStream),
            callbackManager: {
                handleNewToken: onTokenStream,
            },
        }),
        { prompt: QA_PROMPT }
    )

    return new ChatVectorDBQAChain({
        vectorstore,
        combineDocumentsChain: docChain,
        questionGeneratorChain: questionGenerator,
    })
}
