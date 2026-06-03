"""
https://github.com/nate-hunter/hs-recipes-api-2605.git
"""

import asyncio
import dotenv
import os
from typing import Any

from github import Auth, Github
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv()

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

github_token = os.getenv("GITHUB_TOKEN")
git = Github(auth=Auth.Token(github_token)) if github_token else None

repo_url = "https://github.com/nate-hunter/hs-recipes-api-2605.git"
repo_name = repo_url.split("/")[-1].replace(".git", "")
username = repo_url.split("/")[-2]
full_repo_name = f"{username}/{repo_name}"
repo = None
if git is not None:
    repo = git.get_repo(full_repo_name)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Fetch metadata for a GitHub pull request by its number.

    Use when you need PR context such as the author, title, description,
    diff URL, state, or commit SHAs. Pass the integer PR number from the
    user's question (e.g. 4 for "PR number 4").

    Returns a dict with keys: user, title, body, diff_url, state, commit_shas.
    """
    pull_request = repo.get_pull(pr_number)

    commit_shas = []
    for commit in pull_request.get_commits():
        commit_shas.append(commit.sha)

    return {
        "user": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "commit_shas": commit_shas,
    }


get_pr_details_tool = FunctionTool.from_defaults(get_pr_details)


def get_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """Fetch changed files and diffs for a specific commit SHA.

    Use after get_pr_details when you need to see which files changed in a
    commit. Pass a commit SHA from the commit_sha list returned by get_pr_details.

    Returns a list of dicts, each with: filename, status, additions,
    deletions, changes, patch.
    """
    commit = repo.get_commit(commit_sha)
    changed_files: list[dict[str, Any]] = []
    for file in commit.files:
        changed_files.append(
            {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch,
            }
        )
    return changed_files


get_commit_details_tool = FunctionTool.from_defaults(get_commit_details)


def get_file_contents(file_path: str) -> str:
    """Fetch the full contents of a file from the repository.

    Use when the user asks to read or inspect a specific file. Pass the
    repository-relative path (e.g. "app/models.py" or "README.md").

    Returns the file content as a UTF-8 string.
    """
    return repo.get_contents(file_path).decoded_content.decode("utf-8")


get_file_contents_tool = FunctionTool.from_defaults(get_file_contents)


async def add_context_to_state(context: str, ctx: Context) -> str:
    """Store the gathered context summary in the shared workflow state.

    You MUST call this tool after gathering all PR context and BEFORE handing off
    to CommentorAgent. Pass the full context summary as a string; it will be
    stored under 'gathered_contexts' so the CommentorAgent can access it.
    """
    await ctx.store.set("gathered_contexts", context)
    return "Context saved to state."


add_context_to_state_tool = FunctionTool.from_defaults(add_context_to_state)


async def add_comment_to_state(draft_comment: str, ctx: Context) -> str:
    """Store the draft PR review comment in the shared workflow state.

    You MUST call this tool after writing the review and BEFORE handing off to
    ReviewAndPostingAgent. Pass the full comment as a string; it will be stored
    under 'draft_comment' for the reviewer agent to access later.
    """
    await ctx.store.set("draft_comment", draft_comment)
    return "Draft comment saved to state."


add_comment_to_state_tool = FunctionTool.from_defaults(add_comment_to_state)


async def add_final_review_to_state(final_review: str, ctx: Context) -> str:
    """Store the approved final PR review in the shared workflow state.

    Call this after validating the draft review and before posting to GitHub.
    Pass the full final review text as a string; it will be stored under
    'final_review' for reference after the workflow completes.
    """
    await ctx.store.set("final_review", final_review)
    return "Final review saved to state."


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Post a final review comment on a GitHub pull request.

    Use after the review has passed final checks. Pass the PR number and the
    full markdown review text as the comment body.

    Returns a confirmation message when the review is posted successfully.
    """
    pull_request = repo.get_pull(pr_number)
    pull_request.create_review(body=comment)
    return "Review posted successfully."


post_review_to_github_tool = FunctionTool.from_defaults(post_review_to_github)


COMMENTOR_AGENT_SYSTEM_PROMPT = (
    "You are the commentor agent that writes review comments for pull requests as a human reviewer would. \\n\n"
    "Your workflow:\n"
    "1. Hand off to ContextAgent to gather PR details, changed files, and any needed repo files.\n"
    "2. Draft a ~200-300 word review in markdown format.\n"
    "3. Call add_comment_to_state with the full review text.\n"
    "4. Hand off to ReviewAndPostingAgent — never post or finalize the review yourself.\n\n"
    "Ensure to do the following for a thorough review:\n"
    "- Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.\n"
    "- Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \\n\n"
    "  - What is good about the PR? \\n\n"
    "  - Did the author follow ALL contribution rules? What is missing? \\n\n"
    "  - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \\n\n"
    "  - Are new endpoints documented? - use the diff to determine this. \\n\n"
    "  - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \\n\n"
    "- If you need any additional details, you must hand off to the Context Agent. \\n\n"
    "- You should directly address the author. So your comments should sound like: \\n\n"
    '  "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"\n'
    "- After drafting the review, you MUST call add_comment_to_state with the full review text.\n"
    "- You MUST then hand off to ReviewAndPostingAgent. Do NOT output the review as your final response.\n"
    "- You must hand off to the ReviewAndPostingAgent once you are done drafting a review."
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Drafts a PR review comment, saves it to state, and hands off to ReviewAndPostingAgent for final review and posting.",
    tools=[add_comment_to_state_tool],
    system_prompt=COMMENTOR_AGENT_SYSTEM_PROMPT,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

CONTEXT_AGENT_SYSTEM_PROMPT = (
    "You are the context gathering agent.\n"
    "Your workflow:\n"
    "1. Gather PR details, changed files, and any requested repo files using your tools.\n"
    "2. Call add_context_to_state with the full context summary.\n"
    "3. Hand off to CommentorAgent — do NOT summarize context as your final response.\n\n"
    "When gathering context, you MUST gather \\n:\n"
    "  - The details: author, title, body, diff_url, state, and head_sha; \\n\n"
    "  - Changed files; \\n\n"
    "  - Any requested for files; \\n\n"
    "After calling add_context_to_state, you MUST hand off to CommentorAgent. "
    "Do NOT output the context as your final response."
)

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers PR context, saves it to state, and hands off to CommentorAgent.",
    tools=[
        get_pr_details_tool,
        get_commit_details_tool,
        get_file_contents_tool,
        add_context_to_state_tool,
    ],
    system_prompt=CONTEXT_AGENT_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"],
)

REVIEW_AND_POSTING_AGENT_SYSTEM_PROMPT = (
    "You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.\n"
    "Once a review is generated, you need to run a final check and post it to GitHub.\n"
    "  - The review must: \\n\n"
    "  - Be a ~200-300 word review in markdown format. \\n\n"
    "  - Specify what is good about the PR: \\n\n"
    "  - Did the author follow ALL contribution rules? What is missing? \\n\n"
    "  - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \\n\n"
    "  - Are there notes on whether new endpoints were documented? \\n\n"
    "  - Are there suggestions on which lines could be improved upon? Are these lines quoted? \\n\n"
    "  If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \\n\n"
    "  When you are satisfied, post the review to GitHub."
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft PR comment, requests rewrites if needed, and posts the final review to GitHub.",
    tools=[add_final_review_to_state, post_review_to_github_tool],
    system_prompt=REVIEW_AND_POSTING_AGENT_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)


async def main():
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools:", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    if git is not None:
        git.close()
