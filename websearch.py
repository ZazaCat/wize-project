with st.expander("History"):
                for convo in list(st.session_state.conversations.keys()):
                    col1, col2 = st.columns([8, 2])
                    with col1:
                        if st.button(convo):
                            st.session_state.current_conversation = convo
                            save_and_rerun()
                    with col2:
                        if convo != "New Chat" and st.button("🗑️", key=f"delete_{convo}"):
                            del st.session_state.conversations[convo]
                            if st.session_state.current_conversation == convo:
                                st.session_state.current_conversation = None
                                if not st.session_state.conversations:
                                    create_new_conversation()
                            save_and_rerun()

            with st.expander("Settings"):
                websearch = st.checkbox("Web Search", value=True)

                # Add Site input field
                site = st.text_input("Site", placeholder="stackoverflow.com", key="site_input")

                available_models = list(MODEL_CONTEXT_LIMITS.keys())
                selected_model = st.selectbox("Choose Model", available_models, index=available_models.index("gpt-4o"), disabled=st.session_state.is_processing)
                st.session_state.selected_model = selected_model  # Store the selected model in session state

                temperature = st.slider("Temperature", 0.0, 1.0, 1.0, step=0.1, disabled=st.session_state.is_processing)
                max_tokens = st.slider("Max Tokens", 100, 4096, 4096, step=100, disabled=st.session_state.is_processing)
                top_p = st.slider("Top P", 0.0, 1.0, 1.0, step=0.1, disabled=st.session_state.is_processing)

                system_prompt = st.text_area("System Prompt", "You are a helpful AI assistant.", disabled=st.session_state.is_processing)

                encoding = tiktoken.encoding_for_model(selected_model)

            st.button("Logout", on_click=logout)

        if current_conversation:
            st.title(f"{current_conversation or 'No Chat Selected'}")
            display_chat(current_conversation)

            user_input = st.chat_input("Send a message", disabled=st.session_state.is_processing)

            if user_input and not st.session_state.is_processing:
                model_context_window = MODEL_CONTEXT_LIMITS.get(selected_model, 4096)
                user_input_tokens = count_tokens(user_input, encoding)
                if user_input_tokens > model_context_window:
                    st.toast("Your message is too long.", icon="❌")
                else:
                    st.session_state.is_processing = True

                    # Generate title for the first user message in a new chat
                    if current_conversation == "New Chat":
                        title = generate_title(user_input)
                        st.session_state.conversations[title] = st.session_state.conversations.pop(current_conversation)
                        current_conversation = title
                        st.session_state.current_conversation = title

                    st.session_state.conversations[current_conversation].append({"role": "user", "content": user_input})
                    st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations
                    save_and_rerun()

            if st.session_state.is_processing:
                chatbot = Chatbot()
                chatbot.model = selected_model
                chatbot.temperature = temperature
                chatbot.max_tokens = max_tokens
                chatbot.top_p = top_p
                chatbot.system_prompt = system_prompt

                if st.session_state.conversations[current_conversation]:
                    last_message = st.session_state.conversations[current_conversation][-1]
                    content_to_send = last_message.get("file_contents", last_message["content"])
                else:
                    content_to_send = ""

                generated_query = generate_search_query(content_to_send, st.session_state.conversations[current_conversation])
                if websearch and generated_query and generated_query != "CANCEL_WEBSEARCH":
                    queries = generated_query.strip().split('\n')
                    site = st.session_state.get("site_input", "").strip()
                    if site:
                        queries = [f"site:{site} {query}" for query in queries]
                    for query in queries:
                        clean_query = query.replace(f"site:{site} ", "").strip()
                        st.write(f"Searching 🌐 for {clean_query}")

                with st.chat_message("assistant", avatar="https://i.ibb.co/4PbTLG9/20240531-141431.jpg"):
                    response_placeholder = st.empty()
                    response_text = ""
                    typing_indicator = " |"

                    for chunk in chatbot.send_message(st.session_state.conversations[current_conversation], content_to_send, websearch=websearch):
                        response_text = chunk.replace(typing_indicator, "")  # Temporary display without indicator
                        response_placeholder.markdown(response_text + typing_indicator, unsafe_allow_html=True)

                    response_placeholder.markdown(response_text, unsafe_allow_html=True)

                st.session_state.is_processing = False
                st.session_state.all_conversations[st.session_state.username] = st.session_state.conversations
                save_and_rerun()

# Run the main UI
main_ui()
