// Replace your existing send logic with this fetch example
async function sendMessageToBackend(messageText) {
const resp = await fetch('http://127.0.0.1:5000/chat', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({ message: messageText })
});
const data = await resp.json();
return data; // contains { reply, escalate, confidence, source }
}


// Usage inside your existing chat flow
// const userMsg = ...; const res = await sendMessageToBackend(userMsg); showBotReply(res.reply);