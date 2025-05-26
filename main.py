import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from youtube_transcript_api.proxies import WebshareProxyConfig

st.title("YouTube Video Chat Assistant")
# Prompt for API Key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

st.session_state.openai_api_key = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    value=st.session_state.openai_api_key
)

# Proceed only if API key is entered
if not st.session_state.openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Input for YouTube video ID
video_id = st.text_input("Enter YouTube Video ID", "Gfr50f6ZBvo")

# Language codes and names mapping
LANGUAGE_CODES = {
    "ab": "Abkhazian", "aa": "Afar", "af": "Afrikaans", "ak": "Akan", "sq": "Albanian",
    "am": "Amharic", "ar": "Arabic", "hy": "Armenian", "as": "Assamese", "ay": "Aymara",
    "az": "Azerbaijani", "bn": "Bangla", "ba": "Bashkir", "eu": "Basque", "be": "Belarusian",
    "bho": "Bhojpuri", "bs": "Bosnian", "br": "Breton", "bg": "Bulgarian", "my": "Burmese",
    "ca": "Catalan", "ceb": "Cebuano", "zh-Hans": "Chinese (Simplified)", "zh-Hant": "Chinese (Traditional)",
    "co": "Corsican", "hr": "Croatian", "cs": "Czech", "da": "Danish", "dv": "Divehi",
    "nl": "Dutch", "dz": "Dzongkha", "en": "English", "eo": "Esperanto", "et": "Estonian",
    "ee": "Ewe", "fo": "Faroese", "fj": "Fijian", "fil": "Filipino", "fi": "Finnish",
    "fr": "French", "gaa": "Ga", "gl": "Galician", "lg": "Ganda", "ka": "Georgian",
    "de": "German", "el": "Greek", "gn": "Guarani", "gu": "Gujarati", "ht": "Haitian Creole",
    "ha": "Hausa", "haw": "Hawaiian", "iw": "Hebrew", "hi": "Hindi", "hmn": "Hmong",
    "hu": "Hungarian", "is": "Icelandic", "ig": "Igbo", "id": "Indonesian", "iu": "Inuktitut",
    "ga": "Irish", "it": "Italian", "ja": "Japanese", "jv": "Javanese", "kl": "Kalaallisut",
    "kn": "Kannada", "kk": "Kazakh", "kha": "Khasi", "km": "Khmer", "rw": "Kinyarwanda",
    "ko": "Korean", "kri": "Krio", "ku": "Kurdish", "ky": "Kyrgyz", "lo": "Lao",
    "la": "Latin", "lv": "Latvian", "ln": "Lingala", "lt": "Lithuanian", "lua": "Luba-Lulua",
    "luo": "Luo", "lb": "Luxembourgish", "mk": "Macedonian", "mg": "Malagasy", "ms": "Malay",
    "ml": "Malayalam", "mt": "Maltese", "gv": "Manx", "mi": "Māori", "mr": "Marathi",
    "mn": "Mongolian", "mfe": "Morisyen", "ne": "Nepali", "new": "Newari", "nso": "Northern Sotho",
    "no": "Norwegian", "ny": "Nyanja", "oc": "Occitan", "or": "Odia", "om": "Oromo",
    "os": "Ossetic", "pam": "Pampanga", "ps": "Pashto", "fa": "Persian", "pl": "Polish",
    "pt": "Portuguese", "pt-PT": "Portuguese (Portugal)", "pa": "Punjabi", "qu": "Quechua",
    "ro": "Romanian", "rn": "Rundi", "ru": "Russian", "sm": "Samoan", "sg": "Sango",
    "sa": "Sanskrit", "gd": "Scottish Gaelic", "sr": "Serbian", "crs": "Seselwa Creole French",
    "sn": "Shona", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian",
    "so": "Somali", "st": "Southern Sotho", "es": "Spanish", "su": "Sundanese", "sw": "Swahili",
    "ss": "Swati", "sv": "Swedish", "tg": "Tajik", "ta": "Tamil", "tt": "Tatar",
    "te": "Telugu", "th": "Thai", "ti": "Tigrinya", "to": "Tongan", "ts": "Tsonga",
    "uk": "Ukrainian", "ur": "Urdu", "ug": "Uyghur", "uz": "Uzbek", "ve": "Venda",
    "vi": "Vietnamese", "war": "Waray", "cy": "Welsh", "fy": "Western Frisian", "wo": "Wolof",
    "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba", "zu": "Zulu"
}

if st.button("Check Available Languages"):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_languages = []
        for transcript in transcript_list:
            lang_code = transcript.language_code
            lang_name = LANGUAGE_CODES.get(lang_code, lang_code)
            available_languages.append(f"{lang_name} ({lang_code})")
        
        st.success("Available languages for this video:")
        for lang in available_languages:
            st.write(f"- {lang}")
    except Exception as e:
        st.error(f"Error checking languages: {str(e)}")

# Language selection
selected_language = st.selectbox(
    "Select Language",
    options=list(LANGUAGE_CODES.keys()),
    format_func=lambda x: f"{LANGUAGE_CODES[x]} ({x})",
    index=list(LANGUAGE_CODES.keys()).index("en")
)

if st.button("Process Video"):
    try:
        with st.spinner("Processing video transcript..."):
            # Get transcript
            def translate(text):
                llm = ChatOpenAI(model="gpt-4", temperature=0.2,openai_api_key=st.session_state.openai_api_key)
                prompt = PromptTemplate(
                    template="Translate the following text to English: {text}",
                    input_variables=["text"]
                )
                parser = StrOutputParser()
                chain= prompt|llm|parser
                return chain.invoke({'text':text})
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[selected_language])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            if selected_language != "en":
                st.info(f"Translating from {LANGUAGE_CODES[selected_language]} to English...")
                transcript = translate(transcript)
            
            # Text splitting
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])
            
            # Embedding
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=st.session_state.openai_api_key)
            
            # Vector store
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
            
            # LLM setup
            llm = ChatOpenAI(model="gpt-4", temperature=0.2, openai_api_key=st.session_state.openai_api_key)
            
            # Prompt template
            prompt = PromptTemplate(
                template="You are a helpful assistant that can answer questions about the video. The video is: {context}. The question is: {question}",
                input_variables=["context", "question"]
            )
            
            parser = StrOutputParser()
            
            # Chain setup
            chain_pl = RunnableParallel({
                'context': retriever | RunnableLambda(lambda x: "\n".join([doc.page_content for doc in x])),
                'question': RunnablePassthrough(),
            })
            
            chain_seq = prompt | llm | parser
            chain = chain_pl | chain_seq
            
            st.success("Video processed successfully! You can now ask questions.")
            
            # Store the chain in session state
            st.session_state['chain'] = chain
            
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Question input
if 'chain' in st.session_state:
    st.subheader("Ask Questions About the Video")
    question = st.text_area("Enter your question here", height=100, placeholder="Type your question about the video content...")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Thinking..."):
                answer = st.session_state['chain'].invoke(question)
                st.markdown("### Answer:")
                st.markdown(answer)
        else:
            st.warning("Please enter a question first.") 