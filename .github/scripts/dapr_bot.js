// List of owner who can control dapr-bot workflow
// IMPORTANT: Make sure usernames are lower-cased
const owners = [
    'yaron2',
    'cyb3rward0g'
]

const docsIssueBodyTpl = (
    issueNumber
) => `This issue was automatically created by \
[Dapr Bot](https://github.com/dapr/dapr-agents/blob/master/.github/workflows/dapr-bot.yml) because a \"docs-needed\" label \
was added to dapr/dapr#${issueNumber}. \n\n\
TODO: Add more details as per [this template](.github/ISSUE_TEMPLATE/new-content-needed.md).`

module.exports = async ({ github, context }) => {
    if (
        context.eventName == 'issue_comment' &&
        context.payload.action == 'created'
    ) {
        await handleIssueCommentCreate({ github, context })
    } else if (
        context.eventName == 'issues' &&
        context.payload.action == 'labeled'
    ) {
        await handleIssueLabeled({ github, context })
    } else {
        console.log(`[main] event ${context.eventName} not supported, exiting.`)
    }
}

/**
 * Handle issue comment create event.
 */
async function handleIssueCommentCreate({ github, context }) {
    const payload = context.payload
    const issue = context.issue
    const username = context.actor.toLowerCase()
    const isFromPulls = !!payload.issue.pull_request
    const commentBody = ((payload.comment.body || '') + '').trim()
    console.log(`    Issue(owner/repo/number): ${issue.owner}/${issue.repo}/${issue.number} 
        Actor(current username / id): ${username} / ${payload.comment.user.id}
        CommentID: ${payload.comment.id}
        CreatedAt: ${payload.comment.created_at}`
    )

    if (!commentBody || !commentBody.startsWith('/')) {
        // Not a command
        return
    }

    const commandParts = commentBody.split(/\s+/)
    const command = commandParts.shift()
    console.log(`    Command: ${command}`)

    // Commands that can be executed by anyone.
    if (command == '/assign') {
        await cmdAssign(github, issue, username, isFromPulls)
        return
    }

    // Commands that can only be executed by owners.
    if (!owners.includes(username)) {
        console.log(
            `[handleIssueCommentCreate] user ${username} is not an owner, exiting.`
        )
        await commentUserNotAllowed(github, issue, username)
        return
    }

    switch (command) {
        case '/make-me-laugh':
            await cmdMakeMeLaugh(github, issue)
            break
        // TODO: add more in future. Ref: https://github.com/dapr/dapr/blob/master/.github/scripts/dapr_bot.js#L99
        default:
            console.log(
                `[handleIssueCommentCreate] command ${command} not found, exiting.`
            )
            break
    }
}

/**
 * Handle issue labeled event.
 */
async function handleIssueLabeled({ github, context }) {
    const payload = context.payload
    const label = payload.label.name
    const issueNumber = payload.issue.number

    // This should not run in forks.
    if (context.repo.owner !== 'dapr') {
        console.log('[handleIssueLabeled] not running in dapr repo, exiting.')
        return
    }

    // Authorization is not required here because it's triggered by an issue label event.
    // Only authorized users can add labels to issues.
    if (label == 'docs-needed') {
        // Open a new issue
        await github.rest.issues.create({
            owner: 'dapr',
            repo: 'docs',
            title: `New content needed for dapr/dapr#${issueNumber}`,
            labels: ['content/missing-information', 'created-by/dapr-bot'],
            body: docsIssueBodyTpl(issueNumber),
        })
    } else {
        console.log(
            `[handleIssueLabeled] label ${label} not supported, exiting.`
        )
    }
}

/**
 * Assign the issue to the user who commented.
 * @param {*} github GitHub object reference
 * @param {*} issue GitHub issue object
 * @param {string} username GitHub user who commented
 * @param {boolean} isFromPulls is the workflow triggered by a pull request?
 */
async function cmdAssign(github, issue, username, isFromPulls) {
    if (isFromPulls) {
        console.log(
            '[cmdAssign] pull requests unsupported, skipping command execution.'
        )
        return
    } else if (issue.assignees && issue.assignees.length !== 0) {
        console.log(
            '[cmdAssign] issue already has assignees, skipping command execution.'
        )
        return
    }

    await github.rest.issues.addAssignees({
        owner: issue.owner,
        repo: issue.repo,
        issue_number: issue.number,
        assignees: [username],
    })
}

/**
 * Comment a funny joke.
 * @param {*} github GitHub object reference
 * @param {*} issue GitHub issue object
 */
async function cmdMakeMeLaugh(github, issue) {
    const result = await github.request(
        'https://official-joke-api.appspot.com/random_joke'
    )
    jokedata = result.data
    joke = 'I have a bad feeling about this.'
    if (jokedata && jokedata.setup && jokedata.punchline) {
        joke = `${jokedata.setup} - ${jokedata.punchline}`
    }

    await github.rest.issues.createComment({
        owner: issue.owner,
        repo: issue.repo,
        issue_number: issue.number,
        body: joke,
    })
}

/**
 * Sends a comment when the user who tried triggering the bot action is not allowed to do so.
 * @param {*} github GitHub object reference
 * @param {*} issue GitHub issue object
 * @param {string} username GitHub user who commented
 */
async function commentUserNotAllowed(github, issue, username) {
    await github.rest.issues.createComment({
        owner: issue.owner,
        repo: issue.repo,
        issue_number: issue.number,
        body: `👋 @${username}, my apologies but I can't perform this action for you because your username is not in the allowlist in the file ${'`.github/scripts/dapr_bot.js`'}.`,
    })
}