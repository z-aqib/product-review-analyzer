D3 Guardrails & Safety Mechanisms
=================================

This document describes the guardrails and safety mechanisms implemented in our RAG (Retrieval-Augmented Generation) pipeline. These guardrails ensure that both user inputs and LLM outputs are monitored for safety, compliance, and quality.

1\. Overview
------------

Our pipeline consists of three main stages:

1.  **ML Recommender** – Provides candidate items for the user.

2.  **RAG Retrieval** – Retrieves relevant documents and context.

3.  **LLM Advisor** – Generates the final user-facing answer.


Guardrails are applied at **two critical points**:

*   **Input validation**: Ensures the user query is safe before being sent to the RAG or LLM.

*   **Output moderation**: Ensures the generated text does not contain unsafe or inappropriate content.


All guardrail events are logged to our monitoring system for auditability and alerting.

2\. Input Validation Guardrails
-------------------------------

Implemented in src/guards/policy.py via validate\_input\_query().

### 2.1 Checks Performed

*   **Prompt Injection Detection**: Blocks malicious instructions that attempt to override system behavior.

    *   Patterns include phrases like:

        *   "ignore previous instructions"

        *   "pretend you are not bound by"

        *   "act as an unfiltered model"

*   **PII Detection**: Flags potential sensitive information such as:

    *   Emails

    *   Phone numbers


### 2.2 Example Usage

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from guards.policy import validate_input_query, GuardrailViolation  try:      report = validate_input_query(user_query)      print("Input safe:", report)  except GuardrailViolation as e:      print("Guardrail triggered:", e.kind, e.details)   `

### 2.3 Action

*   Hard fail on prompt injection: Query is rejected and logged.

*   Soft fail on PII: Query is flagged but may still be processed depending on policy.


3\. Output Moderation Guardrails
--------------------------------

Implemented in src/guards/policy.py via moderate\_output\_text().

### 3.1 Checks Performed

*   **Toxicity Filter**: Detects unsafe or harmful language in the generated response.

    *   Example keywords: "kill", "hate you", "stupid", "idiot"


### 3.2 Example Usage

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from guards.policy import moderate_output_text, GuardrailViolation  try:      safe_output = moderate_output_text(llm_text)      print("Moderated output:", safe_output["text"])  except GuardrailViolation as e:      print("Guardrail triggered:", e.kind, e.details)   `

### 3.3 Action

*   Soft fail: Output is sanitized and replaced with a safe fallback response.

*   Hard fail: Logged in the monitoring system for review.


4\. Integration with RAG Pipeline
---------------------------------

The guardrails are integrated directly into the RAG pipeline in pipeline.py and rag.py as follows:

1.  **Before RAG Processing**

    *   validate\_input\_query(user\_query) is called to ensure the user query is safe.

    *   If the input fails validation, processing is aborted, and the event is logged.

2.  **After LLM Generation**

    *   moderate\_output\_text(final\_answer) is called on the LLM’s output.

    *   Unsafe outputs trigger a guardrail event, and the response is sanitized or blocked.


### 4.1 Pipeline Flow Diagram

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   USER QUERY      │      ▼  Input Validation (Guardrails) ──► Reject / Flag PII      │      ▼  RAG Retrieval & LLM Advisor      │      ▼  Output Moderation (Guardrails) ──► Sanitize / Block toxic content      │      ▼  FINAL RESPONSE   `

5\. Logging & Monitoring
------------------------

*   All guardrail events (input violations, output toxicity) are captured with:

    *   kind (type of violation)

    *   message (human-readable explanation)

    *   details (pattern match, keywords, or flags)

*   These events can be logged to Prometheus, Evidently AI, or any centralized monitoring system.

*   Example event:


Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   {      "kind": "input_prompt_injection",      "message": "Query contains prompt-injection style instructions.",      "details": {"pattern": "(?i)ignore previous instructions"}  }   `

6\. Extensibility
-----------------

*   **New Rule Types**: Additional rules for input/output can be added by extending policy.py.

*   **Custom Moderation**: Integrate external moderation APIs or more advanced NLP filters.

*   **RAG-specific Guards**: Filters can be applied to retrieved documents before sending context to the LLM.


This guardrails system ensures that our RAG pipeline operates safely, respects user privacy, and prevents harmful outputs.
