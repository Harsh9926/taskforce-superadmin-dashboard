import type { NextApiRequest, NextApiResponse } from 'next';
import OpenAI from 'openai';
import { ComplianceReport, ImageValidationResult } from '@/lib/dataService';
import { ImageQuestionRule } from '@/lib/aiService';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const IMAGE_MODEL = process.env.OPENAI_IMAGE_MODEL || 'gpt-4.1-mini';

if (!OPENAI_API_KEY) {
  console.warn('OPENAI_API_KEY environment variable is not set. ChatGPT image validation will fall back to a placeholder response.');
}

const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

interface ImageAnalysisRequest {
  report: ComplianceReport;
  questionRules: ImageQuestionRule[];
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  if (!openai) {
    return res.status(500).json({ error: 'ChatGPT API key not configured.' });
  }

  const { report, questionRules }: ImageAnalysisRequest = req.body || {};

  if (!report) {
    return res.status(400).json({ error: 'Missing report payload.' });
  }

  try {
    const prompt = buildImagePrompt(report, questionRules || []);
    const response = await openai.responses.create({
      model: IMAGE_MODEL,
      input: prompt,
      temperature: 0.2
    });

    const text = response.output_text || response.output?.map(block => ('content' in block ? block.content?.map((item: any) => item.text?.value || '').join(' ') : '')).join('\n') || '';
    const analysis = normalizeImageAnalysis(text, report, questionRules || []);
    return res.status(200).json(analysis);
  } catch (error) {
    console.error('Error generating ChatGPT image analysis:', error);
    const fallback: ImageValidationResult = {
      overallFindings: 'ChatGPT could not analyze the images. Please review them manually.',
      questionChecks: [],
      aiModel: `chatgpt:${IMAGE_MODEL}`,
      generatedAt: new Date().toISOString()
    };
    return res.status(200).json(fallback);
  }
}

function buildImagePrompt(report: ComplianceReport, questionRules: ImageQuestionRule[]): string {
  const answers = (report.answers || [])
    .map((answer, index) => {
      const label = answer.questionId || `question_${index + 1}`;
      const detail = answer.description ? ` | notes: ${answer.description}` : '';
      return `- ${label}: ${answer.answer || 'no response'}${detail}`;
    })
    .join('\n') || 'No structured answers were submitted.';

  const questionExpectations = questionRules
    .map(rule => `Aliases: ${rule.ids.join(', ')} | Label: ${rule.label} | Expectation: ${rule.expectation}`)
    .join('\n');

  const photos = collectReportPhotoUrls(report);
  const photoSection = photos.length
    ? photos.map((url, index) => `${index + 1}. ${url}`).join('\n')
    : 'No photo URLs were supplied.';

  return `
You are ChatGPT validating municipal compliance photos. Review every image URL and map them to the inspection questions.

Requirements:
1. For each inspection question, decide if the submitted photos SUPPORT (match), CONTRADICT (mismatch), are MISSING, or are UNCERTAIN.
2. Reference specific photo indices/URLs when describing findings.
3. Provide a JSON response ONLY with this schema:
{
  "overallFindings": "2 sentences summarizing image accuracy",
  "questionChecks": [
    {
      "questionId": "exact question id from report.answers",
      "questionLabel": "friendly label",
      "answer": "inspector answer value",
      "status": "match|mismatch|missing|uncertain",
      "summary": "one sentence referencing the specific photo evidence",
      "photoEvidence": ["url1", "url2"]
    }
  ],
  "aiModel": "${IMAGE_MODEL}"
}

Inspection metadata:
- Report ID: ${report.id || 'N/A'}
- Feeder Point: ${report.feederPointName || 'Unknown'}
- Trip Number: ${report.tripNumber || 'Not specified'}

Question expectations:
${questionExpectations}

Inspector answers:
${answers}

Photo URLs (use these indices in your reasoning):
${photoSection}
`;
}

function normalizeImageAnalysis(
  text: string,
  report: ComplianceReport,
  questionRules: ImageQuestionRule[]
): ImageValidationResult {
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  let parsed: ImageValidationResult | null = null;

  if (jsonMatch) {
    try {
      parsed = JSON.parse(jsonMatch[0]);
    } catch (error) {
      console.warn('Unable to parse ChatGPT image analysis JSON:', error);
    }
  }

  if (!parsed) {
    return {
      overallFindings: `ChatGPT output could not be parsed. Raw response: ${text.trim().slice(0, 400)}...`,
      questionChecks: [],
      aiModel: `chatgpt:${IMAGE_MODEL}`,
      generatedAt: new Date().toISOString()
    };
  }

  const labels = buildQuestionLabelLookup(questionRules);
  const answersMap = new Map<string, string>();
  report.answers?.forEach(answer => {
    if (answer.questionId) {
      answersMap.set(answer.questionId.toLowerCase(), answer.answer || '');
    }
  });

  parsed.questionChecks = (parsed.questionChecks || []).map((check, index) => {
    const key = check.questionId?.toLowerCase() || '';
    return {
      questionId: check.questionId,
      questionLabel: check.questionLabel || labels.get(key),
      answer: check.answer || answersMap.get(key) || '',
      status: normalizeStatus(check.status),
      summary: check.summary,
      photoEvidence: check.photoEvidence || []
    };
  });

  parsed.generatedAt = parsed.generatedAt || new Date().toISOString();
  parsed.aiModel = parsed.aiModel || `chatgpt:${IMAGE_MODEL}`;
  return parsed;
}

function normalizeStatus(status: string): 'match' | 'mismatch' | 'missing' | 'uncertain' {
  const normalized = (status || '').toLowerCase();
  if (normalized.includes('match') && normalized.includes('mis')) {
    return 'uncertain';
  }
  if (normalized.includes('match')) {
    return 'match';
  }
  if (normalized.includes('mismatch') || normalized.includes('contrad') || normalized.includes('fail')) {
    return 'mismatch';
  }
  if (normalized.includes('missing') || normalized.includes('no photo')) {
    return 'missing';
  }
  return 'uncertain';
}

function collectReportPhotoUrls(report: ComplianceReport): string[] {
  const urls = new Set<string>();
  report.attachments?.forEach(attachment => {
    if (attachment.url) {
      urls.add(attachment.url);
    }
  });
  report.answers?.forEach(answer => {
    answer.photos?.forEach(photo => {
      if (photo) {
        urls.add(photo);
      }
    });
  });
  return Array.from(urls);
}

function buildQuestionLabelLookup(questionRules: ImageQuestionRule[]): Map<string, string> {
  const map = new Map<string, string>();
  questionRules.forEach(rule => {
    rule.ids.forEach(id => {
      if (id) {
        map.set(id.toLowerCase(), rule.label);
      }
    });
  });
  return map;
}
