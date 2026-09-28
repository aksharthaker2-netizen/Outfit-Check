const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function analyzeOutfit(file) {
  const formData = new FormData()
  formData.append('file', file)

  let response
  try {
    response = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      body: formData,
    })
  } catch {
    throw new Error('Could not reach the server. Is the backend running?')
  }

  if (!response.ok) {
    let message = 'Something went wrong. Please try again.'
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') message = body.detail
    } catch {
      // no JSON body, keep the generic message
    }
    throw new Error(message)
  }

  return response.json()
}