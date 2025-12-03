import { useEffect, useMemo, useState, useRef } from 'react'
import dynamic from 'next/dynamic'
import {
  FileText,
  Send,
  Download,
  Calendar,
  TrendingUp,
  Users,
  AlertTriangle,
  CheckCircle,
  Clock,
  Bot,
  Sparkles,
  Eye,
  X,
  MapPin,
  User,
  Trash2,
  Image as ImageIcon,
  ZoomIn,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  Grid3X3,
  Maximize2,
  Recycle,
  UserCheck,
  Shirt,
  Truck,
  Building,
  Leaf,
  Droplets,
  BarChart3,
  QrCode,
  Award
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { DataService, ComplianceReport } from '@/lib/dataService'
import { AIService, DailyReportData } from '@/lib/aiService'
import { useAuth } from '@/contexts/AuthContext'
import { SimpleBarChart } from '@/components/charts/SimpleBarChart'
import { QuestionPieChart } from '@/components/charts/QuestionPieChart'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'

interface ReportSummary {
  total: number;
  pending: number;
  approved: number;
  rejected: number;
  actionRequired: number;
}

const StatusPieChart = dynamic(() => import('@/components/charts/StatusPieChart').then(mod => mod.StatusPieChart), {
  ssr: false,
})

const YES_COLOR = '#22c55e'
const NO_COLOR = '#ef4444'
const SUMMARY_POINT_LIMIT = 4
const CATEGORY_COLORS = ['#7c3aed', '#0ea5e9', '#f97316', '#14b8a6', '#ec4899', '#a855f7', '#22d3ee', '#facc15']
const YES_ANSWER_VALUES = new Set(['yes', 'y', 'true', '✔️', '✅'])
const NO_ANSWER_VALUES = new Set(['no', 'n', 'false', '❌', '✖️'])
const HIDDEN_QUESTION_CHART_IDS = new Set(['visible_signboard', 'overall_compliance_rating'])
const SWACH_QUESTION_ID = 'swatch_workers_count'
const SWACH_TRIP_UNKNOWN_KEY = '__swach_trip_unknown__'

const QUESTION_METADATA: Record<string, { label: string; icon: LucideIcon }> = {}

const registerQuestionMetadata = (keys: string | string[], label: string, icon: LucideIcon) => {
  const list = Array.isArray(keys) ? keys : [keys]
  list
    .map(key => key?.trim())
    .filter((key): key is string => Boolean(key))
    .forEach(key => {
      QUESTION_METADATA[key] = { label, icon }
    })
}

registerQuestionMetadata(['scp_area_clean', 'scp', 'scp_area'], 'SCP Area Clean', Sparkles)
registerQuestionMetadata(['waste_segregated', 'wet_dry_waste_segregation'], 'Waste Segregation', Recycle)
registerQuestionMetadata(['staff_present'], 'Staff Presence', Users)
registerQuestionMetadata(['swach_workers_present', 'swach_workers_count', 'swatch_workers_count', 'q10', 'Q10', 'q10_swach_workers_present'], 'Swach Workers Present', UserCheck)
registerQuestionMetadata(['workers_wearing_uniform', 'driver_helper_uniform'], 'Uniform Compliance', Shirt)
registerQuestionMetadata(['collection_team_mixing_waste'], 'Team Mixing Waste', AlertTriangle)
registerQuestionMetadata(['vehicle_separate_compartments'], 'Vehicle Compartments', Truck)
registerQuestionMetadata(['Q1', 'q1', 'q1_zone_name', 'zone_name'], 'Zone Name', MapPin)
registerQuestionMetadata(['Q2', 'q2', 'q2_ward_number', 'ward_number'], 'Ward Number', Grid3X3)
registerQuestionMetadata(['Q3', 'q3', 'q3_sc_point_name', 'sc_point_name', 'sc_point'], 'SC Point Name', Building)
registerQuestionMetadata(['Q4', 'q4', 'q4_feeder_point_clean', 'feeder_point_clean'], 'Feeder Point Clean', Sparkles)
registerQuestionMetadata(['Q5', 'q5', 'q5_surrounding_area_clean', 'surrounding_area_clean', 'surrounding_area_maintained'], 'Surrounding Area Clean', Leaf)
registerQuestionMetadata(['Q6', 'q6', 'q6_drains_stormwater_clean', 'drains_stormwater_clean', 'drains_clean'], 'Drains / Stormwater Clean', Droplets)
registerQuestionMetadata(['Q7', 'q7', 'q7_wet_dry_waste_segregation'], 'Wet/Dry Waste Segregation', Recycle)
registerQuestionMetadata(['Q8', 'q8', 'q8_waste_volume_reasonable', 'waste_volume_reasonable'], 'Waste Volume Reasonable', BarChart3)
registerQuestionMetadata(['Q9', 'q9', 'q9_current_waste_status', 'current_waste_status', 'waste_collection_status'], 'Current Waste Status', Truck)
registerQuestionMetadata(['Q11', 'q11', 'q11_signboard_qr_display', 'signboard_qr_display', 'qr_display', 'visible_signboard'], 'Signboard / QR Display', QrCode)
registerQuestionMetadata(['Q12', 'q12', 'q12_overall_score', 'overall_score', 'overall_compliance_rating'], 'Overall Score', Award)

interface QuestionAnswerAggregate {
  normalized: string;
  label: string;
  name: string;
  value: number;
  color: string;
}

type BinaryResponseType = 'yes' | 'no'

interface QuestionBinaryDetailEntry {
  reportId: string;
  feederPointName: string;
  submittedBy: string;
  timestamp: string;
  remarks?: string;
}

interface QuestionBinaryDetails {
  yes: QuestionBinaryDetailEntry[];
  no: QuestionBinaryDetailEntry[];
}

interface QuestionBinaryDetailSelection {
  questionId: string;
  responseType: BinaryResponseType;
}

interface QuestionBreakdown {
  id: string;
  label: string;
  icon: LucideIcon;
  totalResponses: number;
  yesCount: number;
  noCount: number;
  answers: QuestionAnswerAggregate[];
  binaryDetails: QuestionBinaryDetails;
}

interface MemberQuestionConfig {
  id: string;
  label: string;
  icon: LucideIcon;
  aliases?: string[];
}

interface MemberQuestionAnswerCount {
  normalized: string;
  label: string;
  count: number;
}

interface MemberQuestionRow {
  memberId: string;
  memberName: string;
  teamName?: string;
  totalReports: number;
  answers: Partial<Record<string, MemberQuestionAnswerCount[]>>;
}

interface QuestionAnswerRecord {
  normalized: string;
  label: string;
  count: number;
}

const MEMBER_QUESTION_CONFIG: MemberQuestionConfig[] = [
  {
    id: 'scp_area_clean',
    label: 'SCP Area Clean',
    icon: Sparkles,
    aliases: ['scp', 'scp_area', 'q4', 'q4_feeder_point_clean', 'feeder_point_clean'],
  },
  {
    id: 'surrounding_area_maintained',
    label: 'Surrounding Area Maintained',
    icon: Leaf,
    aliases: ['surrounding_area_clean', 'q5', 'q5_surrounding_area_clean'],
  },
  {
    id: 'drains_clean',
    label: 'Drains / Stormwater Clean',
    icon: Droplets,
    aliases: ['drains_stormwater_clean', 'q6', 'q6_drains_stormwater_clean'],
  },
  {
    id: 'waste_segregated',
    label: 'Waste Segregation',
    icon: Recycle,
    aliases: ['wet_dry_waste_segregation', 'q7', 'q7_wet_dry_waste_segregation'],
  },
  {
    id: 'waste_volume_reasonable',
    label: 'Waste Volume Reasonable',
    icon: BarChart3,
    aliases: ['q8', 'q8_waste_volume_reasonable'],
  },
  {
    id: 'waste_collection_status',
    label: 'Waste Collection Status',
    icon: Truck,
    aliases: ['current_waste_status', 'q9', 'q9_current_waste_status'],
  },
  {
    id: 'swatch_workers_count',
    label: 'Swach Workers Count',
    icon: Users,
    aliases: ['swach_workers_present', 'swach_workers_count', 'q10', 'q10_swach_workers_present', 'staff_present'],
  },
  {
    id: 'visible_signboard',
    label: 'Visible Signboard / QR',
    icon: QrCode,
    aliases: ['signboard_qr_display', 'q11', 'q11_signboard_qr_display', 'qr_display'],
  },
  {
    id: 'overall_compliance_rating',
    label: 'Overall Compliance Rating',
    icon: Award,
    aliases: ['overall_score', 'q12', 'q12_overall_score'],
  },
]

const normalizeAnswerValue = (value: unknown) => {
  if (value === null || value === undefined) return 'unspecified'
  return value
    .toString()
    .trim()
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ') || 'unspecified'
}

const formatAnswerDisplay = (rawValue: unknown, normalized: string) => {
  if (normalized === 'unspecified') return 'Not Provided'
  if (normalized === 'yes') return 'Yes'
  if (normalized === 'no') return 'No'

  const raw = rawValue?.toString().trim()
  if (raw) return raw

  return normalized
    .split(' ')
    .map(part => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(' ')
}

const buildSummaryPoints = (summary: string | null): string[] => {
  if (!summary) return []

  const cleanedLines = summary
    .split(/\n+/)
    .map(line => line.replace(/^[\d\.\-\)\u2022•]+\s*/, '').trim())
    .filter(Boolean)

  const uniqueLines: string[] = []
  const seen = new Set<string>()

  for (const line of cleanedLines) {
    const key = line.toLowerCase()
    if (!seen.has(key)) {
      seen.add(key)
      uniqueLines.push(line)
    }
    if (uniqueLines.length >= SUMMARY_POINT_LIMIT) break
  }

  return uniqueLines
}

const getAnswerColor = (normalized: string, index: number) => {
  if (YES_ANSWER_VALUES.has(normalized)) return YES_COLOR
  if (NO_ANSWER_VALUES.has(normalized)) return NO_COLOR
  return CATEGORY_COLORS[index % CATEGORY_COLORS.length]
}

const slugifyQuestionId = (value: string) => {
  if (!value) return ''
  return value
    .toString()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

const sanitizeFilenameSegment = (value: string, fallback = 'value') => {
  if (!value) return fallback
  const sanitized = value
    .toString()
    .trim()
    .replace(/\s+/g, '_')
    .replace(/[^a-zA-Z0-9_-]/g, '')
  return sanitized || fallback
}

const ZONE_QUESTION_KEYS = ['Q1', 'q1', 'q1_zone_name', 'zone_name']
const ZONE_QUESTION_SLUGS = new Set(
  ZONE_QUESTION_KEYS.map(key => slugifyQuestionId(key)).filter((key): key is string => Boolean(key))
)

const getReportZoneInfo = (report: ComplianceReport) => {
  if (!report.answers?.length) return null
  const zoneAnswer = report.answers.find(answer => {
    if (!answer.questionId) return false
    const slug = slugifyQuestionId(answer.questionId)
    return ZONE_QUESTION_SLUGS.has(slug)
  })
  if (!zoneAnswer) return null
  const normalized = normalizeAnswerValue(zoneAnswer.answer)
  const label = formatAnswerDisplay(zoneAnswer.answer, normalized)
  return { normalized, label }
}

const MEMBER_QUESTION_ALIAS_MAP = new Map<string, string>()
MEMBER_QUESTION_CONFIG.forEach(question => {
  const keys = [question.id, ...(question.aliases ?? [])]
  keys
    .map(key => slugifyQuestionId(key))
    .filter(Boolean)
    .forEach(slug => {
      MEMBER_QUESTION_ALIAS_MAP.set(slug, question.id)
    })
})

const getReportMemberKey = (report: ComplianceReport) => report.userId || report.userName || report.id
const formatReportDate = (report: ComplianceReport) => {
  try {
    if (report.submittedAt?.toDate) {
      return report.submittedAt.toDate().toLocaleString()
    }
    const parsed = new Date(report.submittedAt)
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toLocaleString()
    }
  } catch {
    // ignore parsing errors
  }
  return '—'
}

const buildMemberQuestionStats = (reports: ComplianceReport[]): MemberQuestionRow[] => {
  if (!reports?.length) return []

  const memberMap = new Map<string, MemberQuestionRow>()

  reports.forEach(report => {
    const memberKey = getReportMemberKey(report)
    if (!memberKey) return

    if (!memberMap.has(memberKey)) {
      memberMap.set(memberKey, {
        memberId: memberKey,
        memberName: report.userName || 'Unassigned Member',
        teamName: report.teamName,
        totalReports: 0,
        answers: {},
      })
    }

    const memberSummary = memberMap.get(memberKey)!
    memberSummary.totalReports += 1

    report.answers?.forEach(answer => {
      if (!answer.questionId) return
      const questionSlug = slugifyQuestionId(answer.questionId)
      const canonicalId = MEMBER_QUESTION_ALIAS_MAP.get(questionSlug)
      if (!canonicalId) return

      const normalizedAnswer = normalizeAnswerValue(answer.answer)
      const displayLabel = formatAnswerDisplay(answer.answer, normalizedAnswer)

      if (!memberSummary.answers[canonicalId]) {
        memberSummary.answers[canonicalId] = []
      }

      const bucket = memberSummary.answers[canonicalId]!
      const existing = bucket.find(entry => entry.normalized === normalizedAnswer)
      if (existing) {
        existing.count += 1
      } else {
        bucket.push({ normalized: normalizedAnswer, label: displayLabel, count: 1 })
      }
    })
  })

  return Array.from(memberMap.values()).sort((a, b) => {
    if (b.totalReports !== a.totalReports) return b.totalReports - a.totalReports
    return a.memberName.localeCompare(b.memberName)
  })
}

function MemberAnswerCell({ answers }: { answers?: MemberQuestionAnswerCount[] }) {
  if (!answers || answers.length === 0) {
    return <span className="text-xs text-gray-400">—</span>
  }

  const total = answers.reduce((sum, entry) => sum + entry.count, 0)
  const yesCount = answers
    .filter(entry => YES_ANSWER_VALUES.has(entry.normalized))
    .reduce((sum, entry) => sum + entry.count, 0)
  const noCount = answers
    .filter(entry => NO_ANSWER_VALUES.has(entry.normalized))
    .reduce((sum, entry) => sum + entry.count, 0)

  if (yesCount + noCount > 0) {
    const yesPercent = Math.round((yesCount / total) * 100) || 0
    const noPercent = Math.round((noCount / total) * 100) || 0
    return (
      <div className="space-y-1 text-[11px] leading-4">
        <div className="flex items-center justify-between text-emerald-600">
          <span>Yes</span>
          <span className="font-semibold">{yesPercent}%</span>
        </div>
        <div className="flex items-center justify-between text-rose-600">
          <span>No</span>
          <span className="font-semibold">{noPercent}%</span>
        </div>
      </div>
    )
  }

  const sorted = [...answers].sort((a, b) => b.count - a.count)
  return (
    <div className="space-y-1 text-[11px] leading-4 text-gray-700">
      {sorted.slice(0, 2).map(entry => (
        <div key={`${entry.normalized}-${entry.label}`} className="flex items-center justify-between gap-2">
          <span className="truncate">{entry.label}</span>
          <span className="font-semibold text-gray-900">{entry.count}</span>
        </div>
      ))}
      {sorted.length > 2 && (
        <p className="text-[10px] text-gray-500">+ {sorted.length - 2} more</p>
      )}
    </div>
  )
}

const reportHasSwachAnswer = (report: ComplianceReport) => {
  return (
    report.answers?.some(answer => {
      if (!answer.questionId) return false
      const questionSlug = slugifyQuestionId(answer.questionId)
      const canonicalId = MEMBER_QUESTION_ALIAS_MAP.get(questionSlug)
      return canonicalId === SWACH_QUESTION_ID
    }) ?? false
  )
}

const getTripKey = (tripNumber: ComplianceReport['tripNumber']) => {
  if (tripNumber === null || tripNumber === undefined) {
    return SWACH_TRIP_UNKNOWN_KEY
  }
  const normalized = `${tripNumber}`.trim()
  return normalized || SWACH_TRIP_UNKNOWN_KEY
}

const getTripLabel = (tripKey: string) => {
  return tripKey === SWACH_TRIP_UNKNOWN_KEY ? 'Unspecified Trip' : `Trip ${tripKey}`
}

const buildSwachExcelRows = (reports: ComplianceReport[]): Array<Record<string, string | number>> => {
  if (!reports?.length) return []

  const feederMap = new Map<string, Map<string, number>>()
  const tripKeys = new Set<string>()

  reports.forEach(report => {
    if (!reportHasSwachAnswer(report)) return

    const feederName = report.feederPointName?.trim() || 'Unknown Feeder Point'
    const tripKey = getTripKey(report.tripNumber)

    tripKeys.add(tripKey)

    if (!feederMap.has(feederName)) {
      feederMap.set(feederName, new Map())
    }

    const tripCounts = feederMap.get(feederName)!
    tripCounts.set(tripKey, (tripCounts.get(tripKey) || 0) + 1)
  })

  if (!feederMap.size) {
    return []
  }

  const sortedTripKeys = Array.from(tripKeys).sort((a, b) => {
    if (a === SWACH_TRIP_UNKNOWN_KEY) return 1
    if (b === SWACH_TRIP_UNKNOWN_KEY) return -1
    return Number(a) - Number(b)
  })

  return Array.from(feederMap.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([feederName, tripCounts]) => {
      const row: Record<string, string | number> = { 'Feeder Point': feederName }
      sortedTripKeys.forEach(tripKey => {
        row[getTripLabel(tripKey)] = tripCounts.get(tripKey) ?? 0
      })
      if (!sortedTripKeys.length) {
        row['Unspecified Trip'] = tripCounts.get(SWACH_TRIP_UNKNOWN_KEY) ?? 0
      }
      return row
    })
}

const buildQuestionBreakdowns = (reports: ComplianceReport[]): QuestionBreakdown[] => {
  if (!reports?.length) return []

  const counts = new Map<string, Map<string, QuestionAnswerRecord>>()
  const binaryDetailsMap = new Map<string, QuestionBinaryDetails>()
  const getBinaryDetailsBucket = (questionId: string) => {
    if (!binaryDetailsMap.has(questionId)) {
      binaryDetailsMap.set(questionId, { yes: [], no: [] })
    }
    return binaryDetailsMap.get(questionId)!
  }

  reports.forEach(report => {
    report.answers?.forEach(answer => {
      if (!answer.questionId) return
      const questionSlug = slugifyQuestionId(answer.questionId)
      const canonicalId = MEMBER_QUESTION_ALIAS_MAP.get(questionSlug)
      if (!canonicalId) return

      const normalized = normalizeAnswerValue(answer.answer)
      const label = formatAnswerDisplay(answer.answer, normalized)

      if (!counts.has(canonicalId)) {
        counts.set(canonicalId, new Map())
      }

      const questionMap = counts.get(canonicalId)!
      if (questionMap.has(normalized)) {
        questionMap.get(normalized)!.count += 1
      } else {
        questionMap.set(normalized, {
          normalized,
          label,
          count: 1,
        })
      }

      if (YES_ANSWER_VALUES.has(normalized) || NO_ANSWER_VALUES.has(normalized)) {
        const binaryBucket = getBinaryDetailsBucket(canonicalId)
        const feederPointName = report.feederPointName?.trim() || 'Unknown Feeder Point'
        const submittedBy =
          report.submittedBy?.toString().trim() ||
          report.userName?.toString().trim() ||
          'Unknown Member'
        const answerNotes = typeof answer.notes === 'string' ? answer.notes.trim() : ''
        const adminNotes = typeof report.adminNotes === 'string' ? report.adminNotes.trim() : ''
        const remarks = answerNotes || adminNotes || undefined

        const detailEntry: QuestionBinaryDetailEntry = {
          reportId: report.id,
          feederPointName,
          submittedBy,
          timestamp: formatReportDate(report),
        }

        if (remarks) {
          detailEntry.remarks = remarks
        }

        if (YES_ANSWER_VALUES.has(normalized)) {
          binaryBucket.yes.push(detailEntry)
        } else {
          binaryBucket.no.push(detailEntry)
        }
      }
    })
  })

  return MEMBER_QUESTION_CONFIG.reduce<QuestionBreakdown[]>((acc, question) => {
    const answerMap = counts.get(question.id)
    if (!answerMap) return acc

    const sortedAnswers = Array.from(answerMap.values()).sort((a, b) => b.count - a.count)
    const totalResponses = sortedAnswers.reduce((sum, entry) => sum + entry.count, 0)
    if (!totalResponses) return acc

    const binaryDetails = binaryDetailsMap.get(question.id) ?? { yes: [], no: [] }
    const yesCount = binaryDetails.yes.length
    const noCount = binaryDetails.no.length

    const answers = sortedAnswers.map((entry, index) => ({
      normalized: entry.normalized,
      label: entry.label,
      name: entry.label,
      value: entry.count,
      color: getAnswerColor(entry.normalized, index),
    }))

    acc.push({
      id: question.id,
      label: question.label,
      icon: question.icon,
      totalResponses,
      yesCount,
      noCount,
      answers,
      binaryDetails,
    })

    return acc
  }, [])
}

type WhatsappDescriptorKey = 'excellent' | 'strong' | 'moderate' | 'attention' | 'critical'

interface WhatsappCategoryDetailArgs {
  positive: number
  negative: number
  breakdown: QuestionBreakdown
}

interface WhatsappCategoryConfig {
  id: string
  label: string
  narrativeLabel?: string
  positiveValues: string[]
  negativeValues?: string[]
  positiveLabel: string
  negativeLabel: string
  descriptorOverrides?: Partial<Record<WhatsappDescriptorKey, string>>
  detailFormatter?: (args: WhatsappCategoryDetailArgs) => string
}

interface WhatsappLineSummary {
  label: string
  narrativeLabel: string
  descriptorKey: WhatsappDescriptorKey
  line: string
}

const WHATSAPP_DESCRIPTOR_SCALE: Array<{ threshold: number; key: WhatsappDescriptorKey; label: string }> = [
  { threshold: 0.8, key: 'excellent', label: 'Very good' },
  { threshold: 0.65, key: 'strong', label: 'Strong' },
  { threshold: 0.5, key: 'moderate', label: 'Moderate' },
  { threshold: 0.35, key: 'attention', label: 'Needs attention' },
  { threshold: 0, key: 'critical', label: 'Critical' },
]

const WHATSAPP_CORE_CATEGORIES: WhatsappCategoryConfig[] = [
  {
    id: 'scp_area_clean',
    label: 'SCP Cleanliness',
    narrativeLabel: 'cleanliness',
    positiveValues: ['yes', 'clean', 'good'],
    negativeValues: ['no', 'unclean', 'not clean', 'dirty'],
    positiveLabel: 'clean',
    negativeLabel: 'unclean',
  },
  {
    id: 'surrounding_area_maintained',
    label: 'Surroundings',
    narrativeLabel: 'surroundings',
    positiveValues: ['maintained', 'clean', 'clear', 'good'],
    negativeValues: ['not maintained', 'not clean', 'dirty', 'unclean', 'clogged'],
    positiveLabel: 'maintained',
    negativeLabel: 'not maintained',
  },
  {
    id: 'drains_clean',
    label: 'Drains',
    narrativeLabel: 'drains',
    positiveValues: ['clean', 'clear', 'desilted'],
    negativeValues: ['dirty', 'clogged', 'not clean', 'blocked'],
    positiveLabel: 'clean',
    negativeLabel: 'not clean',
  },
  {
    id: 'waste_segregated',
    label: 'Segregation',
    narrativeLabel: 'segregation',
    positiveValues: ['yes', 'segregated', 'proper'],
    negativeValues: ['no', 'mixed', 'not segregated'],
    positiveLabel: 'segregated',
    negativeLabel: 'not segregated',
  },
  {
    id: 'waste_volume_reasonable',
    label: 'Waste Volume',
    narrativeLabel: 'waste volume',
    positiveValues: ['reasonable', 'low', 'normal'],
    negativeValues: ['high', 'overflow', 'excess'],
    positiveLabel: 'reasonable',
    negativeLabel: 'high',
  },
]

const sumMatchingAnswers = (breakdown: QuestionBreakdown, keywords?: string[]) => {
  if (!keywords?.length) return 0
  const normalizedKeywords = keywords.map(keyword => keyword.toLowerCase())
  return breakdown.answers.reduce((sum, answer) => {
    if (normalizedKeywords.some(keyword => answer.normalized.includes(keyword))) {
      return sum + answer.value
    }
    return sum
  }, 0)
}

const describePerformanceLevel = (
  positiveCount: number,
  denominator: number,
  overrides?: Partial<Record<WhatsappDescriptorKey, string>>
): { key: WhatsappDescriptorKey; label: string } => {
  const safeDenominator = denominator > 0 ? denominator : 1
  const ratio = positiveCount / safeDenominator
  for (const level of WHATSAPP_DESCRIPTOR_SCALE) {
    if (ratio >= level.threshold) {
      return {
        key: level.key,
        label: overrides?.[level.key] || level.label,
      }
    }
  }
  return { key: 'critical', label: overrides?.critical || 'Critical' }
}

const formatSiteCount = (count: number, description: string) => {
  return `${count} site${count === 1 ? '' : 's'} ${description}`
}

const formatListForSentence = (items: string[]) => {
  if (!items.length) return ''
  if (items.length === 1) return items[0]
  if (items.length === 2) return `${items[0]} and ${items[1]}`
  return `${items.slice(0, -1).join(', ')}, and ${items[items.length - 1]}`
}

const summarizeAnswersForWhatsapp = (breakdown: QuestionBreakdown, maxEntries = 3) => {
  if (!breakdown.answers?.length) {
    const total = breakdown.totalResponses || 0
    return total ? `${total} responses recorded` : 'No responses recorded'
  }

  const topAnswers = breakdown.answers.slice(0, maxEntries)
  const summaryParts = topAnswers.map(answer => `${answer.value} ${answer.label}`)
  const accounted = topAnswers.reduce((sum, answer) => sum + answer.value, 0)
  const remaining = Math.max(breakdown.totalResponses - accounted, 0)
  if (remaining > 0) {
    summaryParts.push(`${remaining} other${remaining === 1 ? '' : 's'}`)
  }

  return `${summaryParts.join(' | ')} (${breakdown.totalResponses} total)`
}

const buildCoreCategorySummaries = (questionBreakdowns: QuestionBreakdown[]): WhatsappLineSummary[] => {
  return WHATSAPP_CORE_CATEGORIES.map(config => {
    const breakdown = questionBreakdowns.find(entry => entry.id === config.id)
    if (!breakdown || !breakdown.totalResponses) {
      return null
    }

    const positive = sumMatchingAnswers(breakdown, config.positiveValues)
    const negativeByKeywords = config.negativeValues?.length ? sumMatchingAnswers(breakdown, config.negativeValues) : 0
    const matched = positive + negativeByKeywords
    const negative =
      matched >= breakdown.totalResponses
        ? negativeByKeywords
        : negativeByKeywords + Math.max(breakdown.totalResponses - matched, 0)
    const denominator = positive + negative
    if (!denominator) {
      return null
    }

    const descriptor = describePerformanceLevel(positive, denominator, config.descriptorOverrides)
    const detailText = config.detailFormatter
      ? config.detailFormatter({ positive, negative, breakdown })
      : summarizeAnswersForWhatsapp(breakdown)

    return {
      label: config.label,
      narrativeLabel: config.narrativeLabel || config.label.toLowerCase(),
      descriptorKey: descriptor.key,
      line: `- ${config.label}: ${descriptor.label} - ${detailText}.`,
    }
  }).filter(Boolean) as WhatsappLineSummary[]
}

const buildWorkerSummary = (breakdown?: QuestionBreakdown | null): WhatsappLineSummary | null => {
  if (!breakdown || !breakdown.totalResponses) {
    return null
  }

  let zero = 0
  let oneToThree = 0
  let fourPlus = 0
  let unspecified = 0

  breakdown.answers.forEach(answer => {
    const normalized = answer.normalized
    const numericMatch = normalized.match(/-?\d+(\.\d+)?/)
    let value: number | null = numericMatch ? Number(numericMatch[0]) : null
    if (Number.isNaN(value)) {
      value = null
    }

    if (value === null) {
      if (normalized.includes('no') || normalized.includes('none') || normalized.includes('nil') || normalized.includes('zero')) {
        value = 0
      } else if (normalized.includes('one')) {
        value = 1
      } else if (normalized.includes('two')) {
        value = 2
      } else if (normalized.includes('three')) {
        value = 3
      }
    }

    if (value === null) {
      unspecified += answer.value
      return
    }

    if (value <= 0) {
      zero += answer.value
    } else if (value <= 3) {
      oneToThree += answer.value
    } else {
      fourPlus += answer.value
    }
  })

  const denominator = zero + oneToThree + fourPlus
  if (!denominator) {
    return null
  }
  const positive = oneToThree + fourPlus
  const descriptor = describePerformanceLevel(positive, denominator, {
    excellent: 'Fully staffed',
    strong: 'Stable coverage',
    moderate: 'Low availability',
    attention: 'Very low availability',
    critical: 'No workforce',
  })

  const detailParts: string[] = []
  if (zero) detailParts.push(formatSiteCount(zero, 'with 0 workers'))
  if (oneToThree) detailParts.push(formatSiteCount(oneToThree, 'with 1-3 workers'))
  if (fourPlus) detailParts.push(formatSiteCount(fourPlus, 'with 4+ workers'))
  if (unspecified) detailParts.push(formatSiteCount(unspecified, 'not reported'))
  const details = detailParts.join('; ') || 'No worker data captured'

  return {
    label: 'Workers',
    narrativeLabel: 'worker availability',
    descriptorKey: descriptor.key,
    line: `- Workers: ${descriptor.label} - ${details}.`,
  }
}

const buildWasteCollectionSummary = (breakdown?: QuestionBreakdown | null): WhatsappLineSummary | null => {
  if (!breakdown || !breakdown.totalResponses) {
    return null
  }

  const positive = sumMatchingAnswers(breakdown, ['collected', 'cleared', 'vehicle left', 'completed', 'done'])
  const negative = sumMatchingAnswers(breakdown, ['waiting', 'pending', 'not', 'delay', 'delayed'])
  const denominator = positive + negative || breakdown.totalResponses
  const descriptor = describePerformanceLevel(positive, denominator, {
    excellent: 'Smooth',
    strong: 'Mostly smooth',
    moderate: 'Mixed',
    attention: 'Delays noted',
    critical: 'Stalled',
  })

  const topAnswers = breakdown.answers.slice(0, 3).map(answer => `${answer.value} ${answer.label.toLowerCase()}`)
  const details = topAnswers.length ? topAnswers.join('; ') : `${breakdown.totalResponses} updates logged`

  return {
    label: 'Waste Collection',
    narrativeLabel: 'waste pickup',
    descriptorKey: descriptor.key,
    line: `- Waste Collection: ${descriptor.label} - ${details}.`,
  }
}

const buildOverallNarrative = (summaries: WhatsappLineSummary[]) => {
  if (!summaries.length) {
    return ''
  }

  const strong = summaries
    .filter(summary => summary.descriptorKey === 'excellent' || summary.descriptorKey === 'strong')
    .map(summary => summary.narrativeLabel)
  const mixed = summaries.filter(summary => summary.descriptorKey === 'moderate').map(summary => summary.narrativeLabel)
  const weak = summaries
    .filter(summary => summary.descriptorKey === 'attention' || summary.descriptorKey === 'critical')
    .map(summary => summary.narrativeLabel)

  const parts: string[] = []
  if (strong.length) {
    parts.push(`Trip performance is strong in ${formatListForSentence(strong)}.`)
  }
  if (mixed.length) {
    parts.push(`${formatListForSentence(mixed)} show mixed results.`)
  }
  if (weak.length) {
    parts.push(`${formatListForSentence(weak)} need attention.`)
  }

  return parts.join(' ')
}

interface WhatsappReportParams {
  reportSummary: ReportSummary | null
  questionBreakdowns: QuestionBreakdown[]
  zoneFilter: 'all' | string
  zoneOptions: Array<{ value: string; label: string }>
  filterTrip: 'all' | 1 | 2 | 3
  useCustomRange: boolean
  startDate: string
  endDate: string
  selectedDate: string
}

const buildWhatsappShortReport = ({
  reportSummary,
  questionBreakdowns,
  zoneFilter,
  zoneOptions,
  filterTrip,
  useCustomRange,
  startDate,
  endDate,
  selectedDate,
}: WhatsappReportParams) => {
  if (!reportSummary || reportSummary.total === 0) {
    return ''
  }

  const zoneLabel =
    zoneFilter === 'all'
      ? 'All Zones'
      : zoneOptions.find(option => option.value === zoneFilter)?.label || `Zone ${zoneFilter}`
  const tripLabel = filterTrip === 'all' ? 'All Trips' : `Trip ${filterTrip}`
  const resolvedStart = startDate || selectedDate
  const resolvedEnd = endDate || resolvedStart
  const dateLabel =
    useCustomRange && resolvedStart && resolvedEnd
      ? resolvedStart === resolvedEnd
        ? resolvedStart
        : `${resolvedStart} to ${resolvedEnd}`
      : selectedDate

  const lines: string[] = [`${zoneLabel} | ${tripLabel} - Concise Report (${dateLabel})`]
  const statusParts: string[] = [`${reportSummary.approved}/${reportSummary.total} Approved`, `${reportSummary.rejected} Rejected`]
  if (reportSummary.actionRequired) {
    statusParts.push(`${reportSummary.actionRequired} Requires Action`)
  }
  if (reportSummary.pending) {
    statusParts.push(`${reportSummary.pending} Pending`)
  }
  lines.push(statusParts.join(' | '))

  const categorySummaries = buildCoreCategorySummaries(questionBreakdowns)
  const workerSummary = buildWorkerSummary(questionBreakdowns.find(entry => entry.id === SWACH_QUESTION_ID))
  const wasteSummary = buildWasteCollectionSummary(
    questionBreakdowns.find(entry => entry.id === 'waste_collection_status')
  )
  const keySummaries: WhatsappLineSummary[] = [
    ...categorySummaries,
    ...(workerSummary ? [workerSummary] : []),
    ...(wasteSummary ? [wasteSummary] : []),
  ]

  if (keySummaries.length) {
    lines.push('', 'Key Field Updates:')
    keySummaries.forEach(summary => lines.push(summary.line))
  }

  const overallNarrative = buildOverallNarrative(
    keySummaries.filter(Boolean) as WhatsappLineSummary[]
  )

  if (overallNarrative) {
    lines.push('', 'Overall:', overallNarrative)
  }

  return lines.join('\n').replace(/\n{3,}/g, '\n\n').trim()
}

export default function DailyReportsPage() {
  const { user } = useAuth()
  const [reports, setReports] = useState<ComplianceReport[]>([])
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0])
  const [useCustomRange, setUseCustomRange] = useState(false)
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0])
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0])
  const [loading, setLoading] = useState(false)
  const [filterStatus, setFilterStatus] = useState<ComplianceReport['status'] | 'all'>('all')
  const [filterTrip, setFilterTrip] = useState<'all' | 1 | 2 | 3>('all')
  const [zoneFilter, setZoneFilter] = useState<'all' | string>('all')
  const [zoneOptions, setZoneOptions] = useState<Array<{ value: string; label: string }>>([])
  const [generatingAI, setGeneratingAI] = useState(false)
  const [selectedReport, setSelectedReport] = useState<ComplianceReport | null>(null)
  const [adminNotes, setAdminNotes] = useState('')
  const [updatingStatus, setUpdatingStatus] = useState(false)
  const [reportSummary, setReportSummary] = useState<ReportSummary | null>(null)
  const [dailyAiDetailedAnalysis, setDailyAiDetailedAnalysis] = useState<string | null>(null)
  const [dailyAiSummary, setDailyAiSummary] = useState<string | null>(null)
  const [selectedAnalysisType, setSelectedAnalysisType] = useState<'detailed' | 'summary' | null>(null)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [imageLoadingStates, setImageLoadingStates] = useState<{ [key: string]: boolean }>({})
  const [imageErrorStates, setImageErrorStates] = useState<{ [key: string]: boolean }>({})
  const [showCollage, setShowCollage] = useState(false)
  const [allImages, setAllImages] = useState<Array<{ url: string, title: string, type: 'answer' | 'attachment' }>>([])
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const summaryRef = useRef<HTMLDivElement>(null);
  const [selectedMemberId, setSelectedMemberId] = useState<string | null>(null)
  const [swachFeederFilter, setSwachFeederFilter] = useState<'all' | string>('all')
  const [swachTripFilter, setSwachTripFilter] = useState<'all' | ComplianceReport['tripNumber']>('all')
  const [binaryDetailSelection, setBinaryDetailSelection] = useState<QuestionBinaryDetailSelection | null>(null)
  const reportMap = useMemo(() => {
    const map = new Map<string, ComplianceReport>()
    reports.forEach(report => {
      if (report.id) {
        map.set(report.id, report)
      }
    })
    return map
  }, [reports])
  
  const dateRangeLabel = useMemo(() => {
    if (useCustomRange && startDate && endDate) {
      const [rangeStart, rangeEnd] = startDate <= endDate ? [startDate, endDate] : [endDate, startDate]
      return sanitizeFilenameSegment(`${rangeStart}_to_${rangeEnd}`, selectedDate)
    }
    return sanitizeFilenameSegment(selectedDate, selectedDate)
  }, [useCustomRange, startDate, endDate, selectedDate])
  const zoneLabelForFilename = useMemo(() => {
    if (zoneFilter === 'all') {
      return 'All_Zones'
    }
    const option = zoneOptions.find(option => option.value === zoneFilter)
    return sanitizeFilenameSegment(option?.label || zoneFilter, 'Zone')
  }, [zoneFilter, zoneOptions])
  const captureSummarySnapshot = async () => {
    if (!summaryRef.current) return null
    const scale = typeof window !== 'undefined' && window.devicePixelRatio ? Math.max(2, window.devicePixelRatio) : 2
    const canvas = await html2canvas(summaryRef.current, {
      scale,
      scrollY: typeof window !== 'undefined' ? -window.scrollY : 0,
      backgroundColor: '#ffffff'
    })
    const imgData = canvas.toDataURL('image/png')
    return { canvas, imgData }
  }
  const captureSummarySections = async () => {
    if (!summaryRef.current) return []
    const targets = summaryRef.current.querySelectorAll<HTMLElement>('[data-pdf-section]')
    const elements = targets.length ? Array.from(targets) : [summaryRef.current]
    const scale = typeof window !== 'undefined' && window.devicePixelRatio ? Math.max(2, window.devicePixelRatio) : 2
    const snapshots: Array<{ imgData: string; width: number; height: number }> = []

    for (const element of elements) {
      const canvas = await html2canvas(element, {
        scale,
        scrollY: typeof window !== 'undefined' ? -window.scrollY : 0,
        backgroundColor: '#ffffff'
      })
      snapshots.push({
        imgData: canvas.toDataURL('image/png'),
        width: canvas.width,
        height: canvas.height,
      })
    }

    return snapshots
  }

  const conciseSummaryPoints = useMemo(() => buildSummaryPoints(dailyAiSummary), [dailyAiSummary])
  const summaryFeederHighlights = useMemo(() => {
    if (!dailyAiSummary || !reports.length) {
      return []
    }

    const feederMap = new Map<
      string,
      {
        feeder: string
        zoneLabel: string
        trips: Map<string, string>
      }
    >()

    reports.forEach(report => {
      const feederName = report.feederPointName?.trim() || 'Unknown Feeder Point'
      const zoneInfo = getReportZoneInfo(report)
      const zoneLabel = zoneInfo?.label || 'Zone Not Provided'
      if (!feederMap.has(feederName)) {
        feederMap.set(feederName, {
          feeder: feederName,
          zoneLabel,
          trips: new Map<string, string>(),
        })
      } else if (zoneInfo?.label) {
        const existing = feederMap.get(feederName)!
        if (existing.zoneLabel === 'Zone Not Provided') {
          existing.zoneLabel = zoneInfo.label
        }
      }

      const detail = feederMap.get(feederName)!
      const tripKey = getTripKey(report.tripNumber)
      detail.trips.set(tripKey, getTripLabel(tripKey))
    })

    const sortTripKeys = (a: string, b: string) => {
      if (a === SWACH_TRIP_UNKNOWN_KEY) return 1
      if (b === SWACH_TRIP_UNKNOWN_KEY) return -1
      const aNum = Number(a)
      const bNum = Number(b)
      if (Number.isNaN(aNum) && Number.isNaN(bNum)) {
        return a.localeCompare(b)
      }
      if (Number.isNaN(aNum)) return 1
      if (Number.isNaN(bNum)) return -1
      return aNum - bNum
    }

    return Array.from(feederMap.values())
      .map(detail => ({
        feeder: detail.feeder,
        zoneLabel: detail.zoneLabel,
        trips: Array.from(detail.trips.entries())
          .sort((a, b) => sortTripKeys(a[0], b[0]))
          .map(([, label]) => label),
      }))
      .sort((a, b) => a.feeder.localeCompare(b.feeder))
  }, [dailyAiSummary, reports])
  const swachFilterOptions = useMemo(() => {
    const feeders = new Set<string>()
    const trips = new Set<ComplianceReport['tripNumber']>()
    reports.forEach(report => {
      if (report.feederPointName) {
        feeders.add(report.feederPointName)
      }
      if (report.tripNumber) {
        trips.add(report.tripNumber)
      }
    })
    return {
      feeders: Array.from(feeders).sort((a, b) => a.localeCompare(b)),
      trips: Array.from(trips).sort((a, b) => Number(a) - Number(b)) as ComplianceReport['tripNumber'][],
    }
  }, [reports])
  const filteredSwachReports = useMemo(() => {
    return reports.filter(report => {
      const matchesFeeder = swachFeederFilter === 'all' || report.feederPointName === swachFeederFilter
      const matchesTrip = swachTripFilter === 'all' || report.tripNumber === swachTripFilter
      return matchesFeeder && matchesTrip
    })
  }, [reports, swachFeederFilter, swachTripFilter])
  const baseQuestionBreakdowns = useMemo(() => buildQuestionBreakdowns(reports), [reports])
  const swachFilteredBreakdown = useMemo(() => {
    if (swachFeederFilter === 'all' && swachTripFilter === 'all') {
      return null
    }
    const breakdown = buildQuestionBreakdowns(filteredSwachReports).find(b => b.id === SWACH_QUESTION_ID)
    if (breakdown) return breakdown
    const meta = MEMBER_QUESTION_CONFIG.find(question => question.id === SWACH_QUESTION_ID)
    if (!meta) return null
    return {
      id: meta.id,
      label: meta.label,
      icon: meta.icon,
      totalResponses: 0,
      yesCount: 0,
      noCount: 0,
      answers: [],
      binaryDetails: { yes: [], no: [] },
    }
  }, [filteredSwachReports, swachFeederFilter, swachTripFilter])
  const swachExcelRows = useMemo(() => buildSwachExcelRows(filteredSwachReports), [filteredSwachReports])
  const questionBreakdowns = useMemo(() => {
    const patched = baseQuestionBreakdowns.map(breakdown => {
      if (breakdown.id === SWACH_QUESTION_ID && swachFilteredBreakdown) {
        return swachFilteredBreakdown
      }
      return breakdown
    })
    if (swachFilteredBreakdown && !patched.some(breakdown => breakdown.id === SWACH_QUESTION_ID)) {
      patched.push(swachFilteredBreakdown)
    }
    return patched.filter(breakdown => !HIDDEN_QUESTION_CHART_IDS.has(breakdown.id))
  }, [baseQuestionBreakdowns, swachFilteredBreakdown])
  const selectedBinaryDetails = useMemo(() => {
    if (!binaryDetailSelection) return null
    const breakdown = questionBreakdowns.find(entry => entry.id === binaryDetailSelection.questionId)
    if (!breakdown) return null
    const entries = breakdown.binaryDetails[binaryDetailSelection.responseType]
    if (!entries.length) return null
    return {
      ...binaryDetailSelection,
      questionLabel: breakdown.label,
      entries,
    }
  }, [binaryDetailSelection, questionBreakdowns])
  useEffect(() => {
    if (binaryDetailSelection && !selectedBinaryDetails) {
      setBinaryDetailSelection(null)
    }
  }, [binaryDetailSelection, selectedBinaryDetails])
  const whatsappReportText = useMemo(
    () =>
      buildWhatsappShortReport({
        reportSummary,
        questionBreakdowns,
        zoneFilter,
        zoneOptions,
        filterTrip,
        useCustomRange,
        startDate,
        endDate,
        selectedDate,
      }),
    [
      reportSummary,
      questionBreakdowns,
      zoneFilter,
      zoneOptions,
      filterTrip,
      useCustomRange,
      startDate,
      endDate,
      selectedDate,
    ]
  )
  const memberQuestionStats = useMemo(() => buildMemberQuestionStats(reports), [reports])
  const selectedMember = useMemo(
    () => memberQuestionStats.find(member => member.memberId === selectedMemberId) || null,
    [memberQuestionStats, selectedMemberId]
  )
  const selectedMemberReports = useMemo(() => {
    if (!selectedMemberId) return []
    return reports
      .filter(report => getReportMemberKey(report) === selectedMemberId)
      .sort((a, b) => {
        const aTime = a.submittedAt?.toDate ? a.submittedAt.toDate().getTime() : new Date(a.submittedAt).getTime()
        const bTime = b.submittedAt?.toDate ? b.submittedAt.toDate().getTime() : new Date(b.submittedAt).getTime()
        return bTime - aTime
      })
  }, [reports, selectedMemberId])
  const selectedMemberFeederPoints = useMemo(() => {
    if (!selectedMemberReports.length) return []
    const names = new Set<string>()
    selectedMemberReports.forEach(report => {
      if (report.feederPointName) {
        names.add(report.feederPointName)
      }
    })
    return Array.from(names).sort((a, b) => a.localeCompare(b))
  }, [selectedMemberReports])
  const handleMemberRowClick = (memberId: string) => {
    setSelectedMemberId(prev => (prev === memberId ? null : memberId))
  }
  const handleBinaryDetailClick = (questionId: string, responseType: BinaryResponseType) => {
    const breakdown = questionBreakdowns.find(entry => entry.id === questionId)
    if (!breakdown) return
    if (!breakdown.binaryDetails[responseType].length) return
    setBinaryDetailSelection({ questionId, responseType })
  }
  const closeBinaryDetailModal = () => setBinaryDetailSelection(null)
  const openReportFromBinaryDetail = (reportId: string) => {
    const report = reportMap.get(reportId)
    if (!report) return
    setSelectedReport(report)
    setBinaryDetailSelection(null)
  }

  const handleDownloadPdf = async () => {
    const sections = await captureSummarySections()
    if (!sections.length) return

    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    const marginX = 10;
    const marginY = 10;
    const contentWidth = pdfWidth - marginX * 2;
    const contentHeight = pdfHeight - marginY * 2;
    const sectionSpacing = 6;
    let currentY = 0;

    sections.forEach((section, index) => {
      let renderWidth = contentWidth;
      let renderHeight = (section.height * renderWidth) / section.width;

      if (renderHeight > contentHeight) {
        const scaleFactor = contentHeight / renderHeight;
        renderWidth *= scaleFactor;
        renderHeight = contentHeight;
      }

      if (currentY > 0 && currentY + renderHeight > contentHeight) {
        pdf.addPage();
        currentY = 0;
      }

      const xPosition = marginX + (contentWidth - renderWidth) / 2;
      const yPosition = marginY + currentY;

      pdf.addImage(section.imgData, 'PNG', xPosition, yPosition, renderWidth, renderHeight);

      currentY += renderHeight + sectionSpacing;

      if (index < sections.length - 1 && currentY > contentHeight) {
        pdf.addPage();
        currentY = 0;
      }
    });

    pdf.save(`Daily_AI_Concise_Summary_${dateRangeLabel}_${zoneLabelForFilename}.pdf`);
  };

  const handleDownloadPpt = async () => {
    const snapshot = await captureSummarySnapshot()
    if (!snapshot) return

    const { default: PptxGenJS } = await import('pptxgenjs')
    const pptx = new PptxGenJS()
    const slide = pptx.addSlide()
    const slideWidth = pptx.presLayout.width
    const slideHeight = pptx.presLayout.height

    slide.addImage({ data: snapshot.imgData, x: 0, y: 0, w: slideWidth, h: slideHeight })
    await pptx.writeFile({ fileName: `Daily_AI_Concise_Summary_${dateRangeLabel}_${zoneLabelForFilename}.pptx` })
  }

  const handleDownloadSwachExcel = async () => {
    if (!swachExcelRows.length) return

    const XLSX = await import('xlsx')
    const worksheet = XLSX.utils.json_to_sheet(swachExcelRows)
    const workbook = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Swach Workers')

    const feederSuffix =
      swachFeederFilter !== 'all' ? `_${slugifyQuestionId(swachFeederFilter)}` : ''
    const zoneSuffix = zoneLabelForFilename ? `_${zoneLabelForFilename}` : ''

    XLSX.writeFile(workbook, `swach-workers-${dateRangeLabel || 'report'}${feederSuffix}${zoneSuffix}.xlsx`)
  }

  const summaryReportPieData = useMemo(() => {
    if (!reportSummary) return []

    return [
      { name: 'Total', value: reportSummary.total, color: '#3b82f6' },
      { name: 'Approved', value: reportSummary.approved, color: '#22c55e' },
      { name: 'Pending', value: reportSummary.pending, color: '#f59e0b' },
    ].filter(item => item.value > 0);
  }, [reportSummary]);

  const statusPieData = useMemo(() => {
    if (!reportSummary) return []

    return [
      { name: 'Pending', value: reportSummary.pending, color: '#f59e0b' },
      { name: 'Approved', value: reportSummary.approved, color: '#22c55e' },
      { name: 'Rejected', value: reportSummary.rejected, color: '#ef4444' },
      { name: 'Action Required', value: reportSummary.actionRequired, color: '#f97316' },
    ]
  }, [reportSummary])

  const statusPieDataFiltered = useMemo(
    () => statusPieData.filter(item => item.value > 0),
    [statusPieData]
  )

  const hasStatusData = statusPieDataFiltered.length > 0

  useEffect(() => {
    if (selectedReport) {
      setImageLoadingStates({})
      setImageErrorStates({})
      return
    }

    setSelectedImage(null)
    setShowCollage(false)
    setAllImages([])
    setCurrentImageIndex(0)
    setImageLoadingStates({})
    setImageErrorStates({})
  }, [selectedReport])

  useEffect(() => {
    if (!selectedMemberId) return
    const hasMember = reports.some(report => getReportMemberKey(report) === selectedMemberId)
    if (!hasMember) {
      setSelectedMemberId(null)
    }
  }, [reports, selectedMemberId])

  useEffect(() => {
    if (swachFeederFilter !== 'all' && !swachFilterOptions.feeders.includes(swachFeederFilter)) {
      setSwachFeederFilter('all')
    }
    if (swachTripFilter !== 'all' && !swachFilterOptions.trips.includes(swachTripFilter)) {
      setSwachTripFilter('all')
    }
  }, [swachFilterOptions, swachFeederFilter, swachTripFilter])

  useEffect(() => {
    if (zoneFilter !== 'all' && !zoneOptions.some(option => option.value === zoneFilter)) {
      setZoneFilter('all')
    }
  }, [zoneOptions, zoneFilter])

  useEffect(() => {
    setLoading(true)
    const unsubscribe = DataService.onComplianceReportsChange(allReports => {
      console.log("All Compliance Reports (real-time):", allReports)
      let filteredReports = allReports.filter(report => {
        const reportDate = new Date(report.submittedAt.toDate()).toISOString().split('T')[0]
        if (useCustomRange && startDate && endDate) {
          const [rangeStart, rangeEnd] = startDate <= endDate ? [startDate, endDate] : [endDate, startDate]
          return reportDate >= rangeStart && reportDate <= rangeEnd
        }

        return reportDate === selectedDate
      })

      const zoneMap = new Map<string, string>()
      filteredReports.forEach(report => {
        const zoneInfo = getReportZoneInfo(report)
        if (zoneInfo && !zoneMap.has(zoneInfo.normalized)) {
          zoneMap.set(zoneInfo.normalized, zoneInfo.label)
        }
      })
      const zoneOptionList = Array.from(zoneMap.entries())
        .map(([value, label]) => ({ value, label }))
        .sort((a, b) => a.label.localeCompare(b.label))
      setZoneOptions(zoneOptionList)

      // Apply status filter
      if (filterStatus !== 'all') {
        filteredReports = filteredReports.filter(report => report.status === filterStatus)
      }

      // Apply trip filter
      if (filterTrip !== 'all') {
        filteredReports = filteredReports.filter(report => report.tripNumber === filterTrip)
      }

      if (zoneFilter !== 'all') {
        filteredReports = filteredReports.filter(report => {
          const zoneInfo = getReportZoneInfo(report)
          return zoneInfo?.normalized === zoneFilter
        })
      }

      console.log("Filtered Compliance Reports (real-time):", filteredReports)
      setReports(filteredReports)

      // Calculate summary
      const summary: ReportSummary = {
        total: filteredReports.length,
        pending: filteredReports.filter(r => r.status === 'pending').length,
        approved: filteredReports.filter(r => r.status === 'approved').length,
        rejected: filteredReports.filter(r => r.status === 'rejected').length,
        actionRequired: filteredReports.filter(r => r.status === 'requires_action').length,
      }
      setReportSummary(summary)

      setLoading(false)
    })

    return () => unsubscribe()
  }, [selectedDate, filterStatus, filterTrip, zoneFilter, useCustomRange, startDate, endDate])

  useEffect(() => {
    if (!useCustomRange) {
      setStartDate(selectedDate)
      setEndDate(selectedDate)
    }
  }, [selectedDate, useCustomRange])

  // Keyboard navigation for image modal
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (!selectedImage) return

      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault()
          navigateImage('prev')
          break
        case 'ArrowRight':
          event.preventDefault()
          navigateImage('next')
          break
        case 'Escape':
          event.preventDefault()
          closeImageModal()
          break
        case 'g':
          if (event.ctrlKey || event.metaKey) {
            event.preventDefault()
            openCollage()
          }
          break
      }
    }

    if (selectedImage) {
      document.addEventListener('keydown', handleKeyPress)
      return () => document.removeEventListener('keydown', handleKeyPress)
    }
  }, [selectedImage, allImages, currentImageIndex])

  // Image handling functions
  const handleImageLoad = (imageUrl: string) => {
    setImageLoadingStates(prev => ({ ...prev, [imageUrl]: false }))
  }

  const handleImageError = (imageUrl: string) => {
    setImageLoadingStates(prev => ({ ...prev, [imageUrl]: false }))
    setImageErrorStates(prev => ({ ...prev, [imageUrl]: true }))
  }

  const openImageModal = (imageUrl: string) => {
    // Collect all images from the current report
    if (selectedReport) {
      const images: Array<{ url: string, title: string, type: 'answer' | 'attachment' }> = []

      // Add images from answers
      selectedReport.answers.forEach((answer, answerIndex) => {
        if (answer.photos) {
          answer.photos.forEach((photoUrl, photoIndex) => {
            images.push({
              url: photoUrl,
              title: `Question ${answer.questionId} - Photo ${photoIndex + 1}`,
              type: 'answer'
            })
          })
        }
      })

      // Add images from attachments
      if (selectedReport.attachments) {
        selectedReport.attachments.forEach((attachment) => {
          if (attachment.type === 'photo') {
            images.push({
              url: attachment.url,
              title: attachment.filename,
              type: 'attachment'
            })
          }
        })
      }

      setAllImages(images)
      const currentIndex = images.findIndex(img => img.url === imageUrl)
      setCurrentImageIndex(currentIndex >= 0 ? currentIndex : 0)
    }

    setImageErrorStates(prev => {
      const updated = { ...prev }
      delete updated[imageUrl]
      return updated
    })

    setImageLoadingStates(prev => ({ ...prev, [imageUrl]: true }))
    setSelectedImage(imageUrl)
  }

  const closeImageModal = () => {
    setSelectedImage(null)
    setShowCollage(false)
  }

  const navigateImage = (direction: 'prev' | 'next') => {
    if (allImages.length === 0) return

    let newIndex = currentImageIndex
    if (direction === 'prev') {
      newIndex = currentImageIndex > 0 ? currentImageIndex - 1 : allImages.length - 1
    } else {
      newIndex = currentImageIndex < allImages.length - 1 ? currentImageIndex + 1 : 0
    }

    setCurrentImageIndex(newIndex)
    setSelectedImage(allImages[newIndex].url)
  }

  const openCollage = () => {
    setShowCollage(true)
  }

  const selectImageFromCollage = (imageUrl: string) => {
    const index = allImages.findIndex(img => img.url === imageUrl)
    setCurrentImageIndex(index >= 0 ? index : 0)
    setSelectedImage(imageUrl)
    setShowCollage(false)
  }

  const handleToggleCustomRange = () => {
    if (useCustomRange) {
      const fallbackDate = startDate || new Date().toISOString().split('T')[0]
      setSelectedDate(fallbackDate)
    } else {
      setStartDate(selectedDate)
      setEndDate(selectedDate)
    }

    setUseCustomRange(!useCustomRange)
  }

  const handleStartDateChange = (value: string) => {
    setStartDate(value)
    if (endDate && value > endDate) {
      setEndDate(value)
    }
  }

  const handleEndDateChange = (value: string) => {
    setEndDate(value)
    if (startDate && value < startDate) {
      setStartDate(value)
    }
  }

  const handleShareWhatsappReport = async () => {
    if (!whatsappReportText) {
      return
    }

    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(whatsappReportText)
      }
    } catch (error) {
      console.warn('Failed to copy WhatsApp short report to clipboard:', error)
    }

    if (typeof window !== 'undefined') {
      const whatsappUrl = `https://wa.me/?text=${encodeURIComponent(whatsappReportText)}`
      window.open(whatsappUrl, '_blank')
    }
  }

  const handleUpdateStatus = async (status: ComplianceReport['status']) => {
    if (!selectedReport || !user) return

    setUpdatingStatus(true)
    try {
      await DataService.updateComplianceReportStatus(
        selectedReport.id,
        status,
        adminNotes,
        user.name
      )

      alert(`Report ${status} successfully`)
      setSelectedReport(null) // Close modal
      setAdminNotes('')
    } catch (error) {
      console.error('❌ Error updating report status:', error)
      alert('Failed to update report status')
    } finally {
      setUpdatingStatus(false)
    }
  }

  const generateDailyAIAnalysis = async (type: 'detailed' | 'summary') => {
    if (!reportSummary) return

    setGeneratingAI(true)
    setSelectedAnalysisType(type)
    try {
      let effectiveDateLabel = selectedDate
      if (useCustomRange && startDate && endDate) {
        const [rangeStart, rangeEnd] = startDate <= endDate ? [startDate, endDate] : [endDate, startDate]
        effectiveDateLabel = `${rangeStart} to ${rangeEnd}`
      }

      const reportData: DailyReportData = {
        date: effectiveDateLabel,
        metrics: {
          totalUsers: 0, // This data is not available from compliance reports
          newRegistrations: 0, // This data is not available from compliance reports
          totalComplaints: reportSummary.total,
          resolvedComplaints: reportSummary.approved,
          activeFeederPoints: 0, // This data is not available from compliance reports
          completedInspections: reportSummary.approved, // Assuming approved reports are completed inspections
        },
        performance: {
          complaintResolutionRate: reportSummary.total > 0
            ? Math.round((reportSummary.approved / reportSummary.total) * 100)
            : 0,
          userGrowth: 0, // Not available
          operationalEfficiency: 0, // Not available
        },
        rawReports: reports // Pass the raw reports for detailed analysis
      }

      const { detailed, summary } = await AIService.generateAnalysis(reportData)

      if (type === 'detailed') {
        setDailyAiDetailedAnalysis(detailed)
        setDailyAiSummary(null) // Clear summary if detailed is requested
      } else {
        setDailyAiSummary(summary)
        setDailyAiDetailedAnalysis(null) // Clear detailed if summary is requested
      }

      alert('🤖 Daily AI Analysis completed successfully!')

    } catch (error) {
      console.error('Error generating daily AI analysis:', error)
      alert('❌ Error generating daily AI analysis. Please try again.')
    } finally {
      setGeneratingAI(false)
    }
  }

  const getStatusColor = (status: ComplianceReport['status']) => {
    switch (status) {
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'approved': return 'bg-green-100 text-green-800';
      case 'rejected': return 'bg-red-100 text-red-800';
      case 'requires_action': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: ComplianceReport['status']) => {
    switch (status) {
      case 'pending': return <Clock className="h-5 w-5 text-yellow-600" />;
      case 'approved': return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'rejected': return <X className="h-5 w-5 text-red-600" />;
      case 'requires_action': return <AlertTriangle className="h-5 w-5 text-orange-600" />;
      default: return <Clock className="h-5 w-5 text-gray-600" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6 scrollable-content">
      {/* Page Header */}
      <div className="border-b border-gray-200 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <FileText className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">Daily Reports</h1>
              <p className="mt-1 text-sm sm:text-base text-gray-600">
                View daily reports from the field.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-end gap-3">
            <label className="flex items-center space-x-2 text-sm text-gray-600">
              <input
                type="checkbox"
                checked={useCustomRange}
                onChange={handleToggleCustomRange}
                className="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span>Custom Range</span>
            </label>

            {useCustomRange ? (
              <div className="flex items-center space-x-2">
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => handleStartDateChange(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <span className="text-sm text-gray-500">to</span>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => handleEndDateChange(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>
            ) : (
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            )}

            <div className="relative">
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value as ComplianceReport['status'] | 'all')}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="all">All Reports</option>
                <option value="pending">Pending</option>
                <option value="approved">Approved</option>
                <option value="rejected">Rejected</option>
                <option value="requires_action">Action Required</option>
              </select>
            </div>

            <div className="relative">
              <select
                value={filterTrip}
                onChange={(e) => setFilterTrip(e.target.value === 'all' ? 'all' : Number(e.target.value) as 1 | 2 | 3)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="all">All Trips</option>
                <option value="1">Trip 1</option>
                <option value="2">Trip 2</option>
                <option value="3">Trip 3</option>
              </select>
            </div>

           <div className="relative">
              <select
                value={zoneFilter}
                onChange={(e) => setZoneFilter(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="all">All Zones</option>
                {zoneOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
         
            
           
          </div>
        </div>
      </div>

      {/* Report Summary */}
      {reportSummary && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="card p-4 text-center">
              <h3 className="text-lg font-semibold text-gray-700">Total Reports</h3>
              <p className="text-3xl font-bold text-blue-600">{reportSummary.total}</p>
            </div>
            <div className="card p-4 text-center">
              <h3 className="text-lg font-semibold text-gray-700">Pending</h3>
              <p className="text-3xl font-bold text-yellow-600">{reportSummary.pending}</p>
            </div>
            <div className="card p-4 text-center">
              <h3 className="text-lg font-semibold text-gray-700">Approved</h3>
              <p className="text-3xl font-bold text-green-600">{reportSummary.approved}</p>
            </div>
            <div className="card p-4 text-center">
              <h3 className="text-lg font-semibold text-gray-700">Rejected</h3>
              <p className="text-3xl font-bold text-red-600">{reportSummary.rejected}</p>
            </div>
          </div>

          <div className="card p-4">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Report Status Distribution</h3>
                <p className="text-sm text-gray-600">
                  Breakdown of report statuses for the selected {useCustomRange ? 'date range' : 'date'}.
                </p>
              </div>
            </div>

            {hasStatusData ? (
              <div className="mt-6 grid grid-cols-1 gap-8 lg:grid-cols-2">
                <StatusPieChart data={statusPieDataFiltered} />
                <div className="space-y-4">
                  {statusPieData.map((item) => (
                    <div key={item.name} className="flex items-center justify-between rounded-lg border border-gray-100 p-4">
                      <div className="flex items-center space-x-3">
                        <span
                          className="h-3 w-3 rounded-full"
                          style={{ backgroundColor: item.color }}
                        ></span>
                        <span className="text-sm font-medium text-gray-700">{item.name}</span>
                      </div>
                      <span className="text-sm font-semibold text-gray-900">{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="mt-6 text-sm text-gray-500">
                No reports available for the selected {useCustomRange ? 'date range.' : 'date.'}
              </p>
            )}
          </div>
        </>
      )}

      {/* Daily AI Analysis */}
      <div className="grid grid-cols-1 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Daily AI Analysis</h3>
          <div className="flex space-x-2 mb-4">
            <button
              onClick={() => generateDailyAIAnalysis('detailed')}
              disabled={generatingAI || !reportSummary || reportSummary.total === 0}
              className="flex-1 btn-secondary flex items-center justify-center space-x-2"
            >
              <Sparkles className="h-4 w-4" />
              <span>{generatingAI && selectedAnalysisType === 'detailed' ? 'Analyzing...' : 'Generate Detailed AI Analysis'}</span>
            </button>
            <button
              onClick={() => generateDailyAIAnalysis('summary')}
              disabled={generatingAI || !reportSummary || reportSummary.total === 0}
              className="flex-1 btn-secondary flex items-center justify-center space-x-2"
            >
              <Sparkles className="h-4 w-4" />
              <span>{generatingAI && selectedAnalysisType === 'summary' ? 'Analyzing...' : 'Generate Concise AI Summary'}</span>
            </button>
            <button
              onClick={handleShareWhatsappReport}
              disabled={!whatsappReportText}
              className="flex-1 btn-secondary flex items-center justify-center space-x-2 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <Send className="h-4 w-4" />
              <span>WhatsApp Short Report</span>
            </button>
          </div>
          {dailyAiDetailedAnalysis && selectedAnalysisType === 'detailed' && (
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <h4 className="text-md font-semibold text-gray-900 mb-2">Detailed Report:</h4>
              <pre className="whitespace-pre-wrap text-sm text-gray-800 font-sans">
                {dailyAiDetailedAnalysis}
              </pre>
              <button
                onClick={() => {
                  if (dailyAiDetailedAnalysis) {
                    const blob = new Blob([dailyAiDetailedAnalysis], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `Daily_AI_Detailed_Analysis_${dateRangeLabel}_${zoneLabelForFilename}.txt`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                  }
                }}
                className="w-full btn-secondary flex items-center justify-center space-x-2 mt-4"
              >
                <Download className="h-4 w-4" />
                <span>Download Detailed AI Report</span>
              </button>
            </div>
          )}
          {dailyAiSummary && selectedAnalysisType === 'summary' && (
            <div ref={summaryRef} className="bg-gray-50 rounded-lg p-4">
              <div data-pdf-section="true">
                <h4 className="text-md font-semibold text-gray-900 mb-2">Concise Summary:</h4>
                {conciseSummaryPoints.length > 0 ? (
                  <ul className="space-y-3 text-sm text-gray-800">
                    {conciseSummaryPoints.map((point, index) => (
                      <li key={`${point}-${index}`} className="flex items-start space-x-2">
                        <Bot className="h-4 w-4 mt-0.5 text-gray-500" />
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <pre className="whitespace-pre-wrap text-sm text-gray-800 font-sans">
                    {dailyAiSummary}
                  </pre>
                )}
              </div>
              {summaryFeederHighlights.length > 0 && (
                <div data-pdf-section="true" className="mt-6">
                  <h4 className="text-md font-semibold text-gray-900 mb-4">Zone & Trip Coverage:</h4>
                  <div className="space-y-4">
                    {Object.entries(
                      summaryFeederHighlights.reduce((acc, detail) => {
                        const key = `Zone: ${detail.zoneLabel} | Trip ${detail.trips.join(', ')}`
                        if (!acc[key]) {
                          acc[key] = []
                        }
                        acc[key].push(detail.feeder)
                        return acc
                      }, {} as Record<string, string[]>)
                    ).sort((a, b) => {
                      const zoneA = parseInt(a[0].match(/Zone: (\d+)/)?.[1] || '0')
                      const zoneB = parseInt(b[0].match(/Zone: (\d+)/)?.[1] || '0')
                      return zoneA - zoneB
                    }).map(([key, feeders]) => (
                      <div key={key}>
                        <p className="font-semibold text-gray-900 mb-2">{key}</p>
                        <ul className="ml-4 space-y-1">
                          {feeders.map((feeder, idx) => (
                            <li key={`${key}-${idx}`} className="text-sm text-gray-700">• {feeder}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {reportSummary && summaryReportPieData.length > 0 && (
                <div data-pdf-section="true" className="mt-6">
                  <h4 className="text-md font-semibold text-gray-900 mb-2">Report Breakdown:</h4>
                  <div className="mt-6 grid grid-cols-1 gap-8 lg:grid-cols-2">
                    <StatusPieChart data={summaryReportPieData} />
                    <div className="space-y-4">
                      {summaryReportPieData.map((item) => (
                        <div key={item.name} className="flex items-center justify-between rounded-lg border border-gray-100 p-4">
                          <div className="flex items-center space-x-3">
                            <span
                              className="h-3 w-3 rounded-full"
                              style={{ backgroundColor: item.color }}
                            ></span>
                            <span className="text-sm font-medium text-gray-700">{item.name}</span>
                          </div>
                          <span className="text-sm font-semibold text-gray-900">{item.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {questionBreakdowns.length > 0 && (
                <div data-pdf-section="true" className="mt-6">
                  <h4 className="text-md font-semibold text-gray-900 mb-2">AI Compliance Snapshot:</h4>
                  <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {questionBreakdowns.map((breakdown) => {
                      const Icon = breakdown.icon
                      const yesFeederCount = breakdown.binaryDetails.yes.length
                      const noFeederCount = breakdown.binaryDetails.no.length
                      const showBinaryCounts = yesFeederCount > 0 || noFeederCount > 0
                      const renderBinaryButton = (
                        responseType: BinaryResponseType,
                        count: number,
                        colorClass: string,
                        symbol: string
                      ) => {
                        const disabled = count === 0
                        return (
                          <button
                            type="button"
                            disabled={disabled}
                            onClick={() => handleBinaryDetailClick(breakdown.id, responseType)}
                            className={`group flex w-full items-center justify-end gap-2 text-right text-xs transition ${
                              disabled ? 'cursor-not-allowed text-gray-400' : 'text-gray-500 hover:text-primary-600'
                            }`}
                          >
                            <span className={`font-semibold ${colorClass}`}>
                              {symbol} {responseType === 'yes' ? 'Yes' : 'No'}: {count} feeder point{count === 1 ? '' : 's'}
                            </span>
                            <span
                              className={`text-[11px] font-medium ${
                                disabled ? 'text-gray-300' : 'text-primary-600 group-hover:underline'
                              }`}
                            >
                              → View List
                            </span>
                          </button>
                        )
                      }
                      return (
                        <div key={`ai-breakdown-${breakdown.id}`} className="rounded-lg border border-gray-100 bg-white/60 p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              <span className="rounded-full bg-gray-100 p-2 text-gray-700">
                                <Icon className="h-4 w-4" />
                              </span>
                              <div>
                                <p className="text-sm font-semibold text-gray-900">{breakdown.label}</p>
                                <p className="text-xs text-gray-500">{breakdown.totalResponses} responses</p>
                              </div>
                            </div>
                            {showBinaryCounts && (
                              <div className="w-full max-w-[240px] text-right">
                                {renderBinaryButton('yes', yesFeederCount, 'text-emerald-600', '✅')}
                                {renderBinaryButton('no', noFeederCount, 'text-rose-600', '❌')}
                              </div>
                            )}
                          </div>
                          {breakdown.id === SWACH_QUESTION_ID && (
                            <div className="mt-3 grid grid-cols-1 gap-3 text-xs text-gray-600 sm:grid-cols-2 md:grid-cols-4 ">
                              <label className="flex flex-col space-y-1">
                                <span className="font-semibold text-gray-700">Feeder Point</span>
                                <select
                                  value={swachFeederFilter}
                                  onChange={(event) => setSwachFeederFilter(event.target.value)}
                                  className="rounded border border-gray-200 bg-white px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-primary-500"
                                >
                                  <option value="all">All feeder points</option>
                                  {swachFilterOptions.feeders.map(feeder => (
                                    <option key={`swach-filter-${feeder}`} value={feeder}>
                                      {feeder}
                                    </option>
                                  ))}
                                </select>
                              </label>
                              <label className="flex flex-col space-y-1">
                                <span className="font-semibold text-gray-700">Trip</span>
                                <select
                                  value={swachTripFilter === 'all' ? 'all' : swachTripFilter.toString()}
                                  onChange={(event) =>
                                    setSwachTripFilter(
                                      event.target.value === 'all'
                                        ? 'all'
                                        : (Number(event.target.value) as ComplianceReport['tripNumber'])
                                    )
                                  }
                                  className="rounded border border-gray-200 bg-white px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-primary-500"
                                >
                                  <option value="all">All trips</option>
                                  {swachFilterOptions.trips.map(trip => (
                                    <option key={`swach-trip-${trip}`} value={trip}>
                                      Trip {trip}
                                    </option>
                                  ))}
                                </select>
                              </label>
                             
                            </div>
                          )}
                          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                            <QuestionPieChart data={breakdown.answers} />
                            <SimpleBarChart data={breakdown.answers} xLabel="Answer Options" yLabel="Responses" />
                          </div>
                          {breakdown.id === SWACH_QUESTION_ID && (
                            <div className="mt-5 flex flex-col gap-3 rounded-lg border border-gray-200 bg-gray-50/80 p-4 text-xs text-gray-600 sm:flex-row sm:items-center sm:justify-between">
                              <div>
                                <p className="text-sm font-semibold text-gray-900">Download Swach Excel report</p>
                                <p>Exports feeder point and trip counts for the filters above.</p>
                              </div>
                              <button
                                onClick={handleDownloadSwachExcel}
                                disabled={!swachExcelRows.length}
                                className="btn-secondary inline-flex items-center justify-center space-x-2 px-4 py-2 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-60"
                              >
                                <Download className="h-3.5 w-3.5" />
                                <span>Download (.xlsx)</span>
                              </button>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
              <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
                <button
                  onClick={handleDownloadPdf}
                  className="btn-secondary flex items-center justify-center space-x-2"
                >
                  <Download className="h-4 w-4" />
                  <span>Download as PDF</span>
                </button>
                <button
                  onClick={handleDownloadPpt}
                  className="btn-secondary flex items-center justify-center space-x-2"
                >
                  <Download className="h-4 w-4" />
                  <span>Download as PPT</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {memberQuestionStats.length > 0 && (
        <div className="card">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Taskforce Member Responses</h3>
              <p className="text-sm text-gray-500">
                Aggregated answers for critical compliance questions captured in today&apos;s reports.
              </p>
            </div>
            <span className="text-sm font-medium text-gray-500">
              {memberQuestionStats.length} member{memberQuestionStats.length === 1 ? '' : 's'}
            </span>
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 text-left text-xs">
              <thead className="bg-gray-50 text-[11px] uppercase tracking-wide text-gray-500">
                <tr>
                  <th className="whitespace-nowrap p-3 font-semibold text-gray-600">Member</th>
                  {MEMBER_QUESTION_CONFIG.map(question => (
                    <th key={question.id} className="whitespace-nowrap p-3 font-semibold text-gray-600">
                      {question.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 bg-white">
                {memberQuestionStats.map(member => {
                  const isSelected = selectedMemberId === member.memberId
                  return (
                  <tr
                    key={member.memberId}
                    className={`align-top transition ${isSelected ? 'bg-blue-50/70 ring-1 ring-blue-200' : 'hover:bg-gray-50'} cursor-pointer`}
                    onClick={() => handleMemberRowClick(member.memberId)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        handleMemberRowClick(member.memberId)
                      }
                    }}
                    role="button"
                    tabIndex={0}
                    aria-pressed={isSelected}
                  >
                    <td className="p-3 align-top">
                      <div className="font-semibold text-gray-900">{member.memberName}</div>
                      {member.teamName && (
                        <p className="text-[11px] text-gray-500">{member.teamName}</p>
                      )}
                      <p className="text-[11px] text-gray-400">
                        {member.totalReports} report{member.totalReports === 1 ? '' : 's'}
                      </p>
                    </td>
                    {MEMBER_QUESTION_CONFIG.map(question => (
                      <td key={`${member.memberId}-${question.id}`} className="p-3 align-top">
                        <MemberAnswerCell answers={member.answers[question.id]} />
                      </td>
                    ))}
                  </tr>
                )})}
              </tbody>
            </table>
          </div>
          {selectedMember && (
            <div className="mt-4 rounded-lg border border-gray-100 bg-gray-50/80 p-4">
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <div>
                  <h4 className="text-md font-semibold text-gray-900">Selected Member: {selectedMember.memberName}</h4>
                  <p className="text-sm text-gray-500">
                    Showing {selectedMemberReports.length} report{selectedMemberReports.length === 1 ? '' : 's'} within the current date filter.
                  </p>
                </div>
                <button
                  className="text-sm font-medium text-blue-600 hover:text-blue-700"
                  onClick={() => setSelectedMemberId(null)}
                >
                  Clear selection
                </button>
              </div>
              <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
                <div className="rounded-lg border border-white/60 bg-white p-4">
                  <h5 className="text-sm font-semibold text-gray-900">Feeder Points</h5>
                  {selectedMemberFeederPoints.length > 0 ? (
                    <ul className="mt-2 space-y-1 text-xs text-gray-600">
                      {selectedMemberFeederPoints.map(point => (
                        <li key={`${selectedMember.memberId}-${point}`} className="flex items-center space-x-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                          <span className="truncate">{point}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-2 text-xs text-gray-500">No feeder points recorded.</p>
                  )}
                </div>
                <div className="lg:col-span-2">
                  {selectedMemberReports.length > 0 ? (
                    <div className="overflow-x-auto rounded-lg border border-white/60 bg-white">
                      <table className="min-w-full divide-y divide-gray-200 text-left text-xs">
                        <thead className="bg-gray-50 text-[11px] uppercase tracking-wide text-gray-500">
                          <tr>
                            <th className="px-3 py-2 font-semibold text-gray-600">Date</th>
                            <th className="px-3 py-2 font-semibold text-gray-600">Feeder Point</th>
                            <th className="px-3 py-2 font-semibold text-gray-600">Status</th>
                            <th className="px-3 py-2 font-semibold text-gray-600">Action</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {selectedMemberReports.map(report => (
                            <tr key={`member-report-${report.id}`} className="bg-white hover:bg-gray-50">
                              <td className="px-3 py-2 text-gray-700">{formatReportDate(report)}</td>
                              <td className="px-3 py-2 text-gray-900">{report.feederPointName || '—'}</td>
                              <td className="px-3 py-2">
                                <span className={`px-2 py-1 text-xs font-semibold ${getStatusColor(report.status)}`}>
                                  {report.status.replace('_', ' ').toUpperCase()}
                                </span>
                              </td>
                              <td className="px-3 py-2 text-gray-600">
                                <button 
                                  onClick={() => setSelectedReport(report)}
                                  className="text-blue-600 hover:text-blue-800 hover:bg-blue-50 p-2 rounded-lg transition-colors"
                                  title="View Report"
                                >
                                  <Eye className="h-5 w-5" />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-xs text-gray-500">No reports available for the selected member.</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Reports List */}
      <div className="space-y-4">
        {reports.length > 0 ? (
          reports.map(report => (
            <div key={report.id} className="card">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-full ${getStatusColor(report.status)}`}>
                    {getStatusIcon(report.status)}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{report.feederPointName} - Trip {report.tripNumber}</h3>
                    <p className="text-sm text-gray-600">Submitted by {report.userName} ({report.teamName})</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="text-sm text-gray-500">
                    {new Date(report.submittedAt.toDate()).toLocaleDateString()}
                  </div>
                  <button onClick={() => setSelectedReport(report)} className="btn-secondary p-2">
                    <Eye className="h-5 w-5" />
                  </button>
                </div>
              </div>
              <div className="mt-4">
                <p className="text-sm text-gray-700">{report.description}</p>
                <div className="flex items-center justify-between mt-3">
                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                    {/* Count photos in answers */}
                    {(() => {
                      const photoCount = report.answers.reduce((count, answer) =>
                        count + (answer.photos ? answer.photos.length : 0), 0
                      )
                      const attachmentCount = report.attachments ? report.attachments.length : 0
                      const totalImages = photoCount + attachmentCount

                      return totalImages > 0 ? (
                        <span className="flex items-center">
                          <ImageIcon className="h-3 w-3 mr-1" />
                          {totalImages} image{totalImages !== 1 ? 's' : ''}
                        </span>
                      ) : null
                    })()}
                    <span className="flex items-center">
                      <User className="h-3 w-3 mr-1" />
                      {report.answers.length} answer{report.answers.length !== 1 ? 's' : ''}
                    </span>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getStatusColor(report.status)}`}>
                    {report.status.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="card text-center py-12">
            <FileText className="mx-auto h-16 w-16 text-gray-400 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No Reports Found</h3>
            <p className="text-gray-600 mb-6">
              No reports were found for the selected date.
            </p>
          </div>
        )}
      </div>

      {/* Feeder Point Detail Modal */}
      {selectedBinaryDetails && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 px-4 py-8">
          <div className="w-full max-w-2xl overflow-hidden rounded-lg bg-white shadow-2xl">
            <div className="flex items-center justify-between border-b border-gray-100 px-6 py-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">AI Compliance Snapshot</p>
                <p className="text-base font-semibold text-gray-900">{selectedBinaryDetails.questionLabel}</p>
                <p className="text-sm text-gray-600">
                  {selectedBinaryDetails.responseType === 'yes' ? '✅ Yes' : '❌ No'} ·{' '}
                  {selectedBinaryDetails.entries.length} feeder point{selectedBinaryDetails.entries.length === 1 ? '' : 's'}
                </p>
              </div>
              <button
                type="button"
                onClick={closeBinaryDetailModal}
                className="rounded-full p-2 text-gray-400 transition hover:bg-gray-100 hover:text-gray-700"
                aria-label="Close feeder point detail modal"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="max-h-[65vh] overflow-y-auto px-6 py-4 text-sm text-gray-700">
              {selectedBinaryDetails.entries.map((entry, index) => (
                <button
                  key={`${entry.reportId}-${index}`}
                  type="button"
                  onClick={() => openReportFromBinaryDetail(entry.reportId)}
                  className="w-full border-b border-gray-100 py-3 text-left transition hover:bg-gray-50 last:border-b-0"
                >
                  <p className="text-base font-semibold text-gray-900">{entry.feederPointName}</p>
                  <p className="text-xs text-gray-500">Timestamp: {entry.timestamp}</p>
                  <p className="mt-1 text-sm text-gray-700">Submitted by: {entry.submittedBy}</p>
                  {entry.remarks && <p className="mt-1 text-sm text-gray-600">Remarks: {entry.remarks}</p>}
                  <p className="mt-1 text-[11px] font-semibold text-primary-600">Click to open report -&gt;</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Report Modal */}
      {selectedReport && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-2xl p-6 w-full max-w-3xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-900">{selectedReport.feederPointName} - Trip {selectedReport.tripNumber}</h2>
              <button onClick={() => setSelectedReport(null)} className="p-2">
                <X className="h-6 w-6 text-gray-600" />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Details</h3>
                  <div className="space-y-2 text-sm text-gray-700">
                    <p><strong>Submitted By:</strong> {selectedReport.userName} ({selectedReport.teamName})</p>
                    <p><strong>Feeder Point:</strong> {selectedReport.feederPointName}</p>
                    <p><strong>Submitted At:</strong> {new Date(selectedReport.submittedAt.toDate()).toLocaleString()}</p>
                    <p><strong>Status:</strong> <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getStatusColor(selectedReport.status)}`}>{selectedReport.status}</span></p>
                    <p><strong>Priority:</strong> <span className={`px-2 py-1 rounded-full text-xs font-semibold ${selectedReport.priority === 'high' ? 'bg-red-100 text-red-800' :
                      selectedReport.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-blue-100 text-blue-800'
                      }`}>{selectedReport.priority}</span></p>
                    <p><strong>Trip Date:</strong> {selectedReport.tripDate}</p>
                    <p><strong>Trip Number:</strong> {selectedReport.tripNumber}</p>
                    <p><strong>Location:</strong> {selectedReport.submittedLocation.address} ({selectedReport.distanceFromFeederPoint.toFixed(2)}m from FP)</p>
                  </div>
                </div>
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Description</h3>
                  <p className="text-sm text-gray-700">{selectedReport.description}</p>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Answers</h3>
                  <div className="space-y-4">
                    {selectedReport.answers.map((answer, index) => (
                      <div key={index} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <p className="text-sm font-medium text-gray-900 mb-1">Q: {answer.questionId}</p>
                        <p className="text-sm text-gray-700 mb-2">A: {answer.answer}</p>
                        {answer.notes && (
                          <p className="text-xs text-gray-600 mb-3 italic">Notes: {answer.notes}</p>
                        )}

                        {/* Display photos for this answer */}
                        {answer.photos && answer.photos.length > 0 && (
                          <div className="mt-3">
                            <p className="text-xs font-medium text-gray-700 mb-2 flex items-center">
                              <ImageIcon className="h-3 w-3 mr-1" />
                              Photos ({answer.photos.length})
                            </p>
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                              {answer.photos.map((photoUrl, photoIndex) => (
                                <div key={photoIndex} className="relative group" onClick={() => openImageModal(photoUrl)}>
                                  {imageLoadingStates[photoUrl] !== false && !imageErrorStates[photoUrl] && (
                                    <div className="absolute inset-0 bg-gray-200 rounded-md flex items-center justify-center z-10">
                                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                                    </div>
                                  )}
                                  {imageErrorStates[photoUrl] ? (
                                    <div className="w-full h-20 bg-gray-200 rounded-md flex items-center justify-center border-2 border-dashed border-gray-300">
                                      <div className="text-center">
                                        <ImageIcon className="h-6 w-6 text-gray-400 mx-auto mb-1" />
                                        <p className="text-xs text-gray-500">Failed to load</p>
                                      </div>
                                    </div>
                                  ) : (
                                    <div className="relative">
                                      <img
                                        src={photoUrl}
                                        alt={`Photo ${photoIndex + 1} for question ${answer.questionId}`}
                                        className="w-full h-20 object-cover rounded-md cursor-pointer hover:opacity-90 transition-opacity"
                                        onLoad={() => handleImageLoad(photoUrl)}
                                        onError={() => handleImageError(photoUrl)}
                                      />
                                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-200 rounded-md flex items-center justify-center">
                                        <ZoomIn className="h-4 w-4 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                                      </div>
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="mt-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center">
                    <ImageIcon className="h-5 w-5 mr-2" />
                    Attachments
                    {selectedReport.attachments && selectedReport.attachments.length > 0 && (
                      <span className="ml-2 text-sm text-gray-500">({selectedReport.attachments.length})</span>
                    )}
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                    {selectedReport.attachments && selectedReport.attachments.length > 0 ? (
                      selectedReport.attachments.map(att => (
                        <div key={att.id} className="relative group" onClick={() => openImageModal(att.url)}>
                          {att.type === 'photo' ? (
                            <>
                              {imageLoadingStates[att.url] !== false && !imageErrorStates[att.url] && (
                                <div className="absolute inset-0 bg-gray-200 rounded-lg flex items-center justify-center z-10">
                                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                                </div>
                              )}
                              {imageErrorStates[att.url] ? (
                                <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center border-2 border-dashed border-gray-300">
                                  <div className="text-center">
                                    <ImageIcon className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                                    <p className="text-xs text-gray-500">Failed to load</p>
                                    <p className="text-xs text-gray-400">{att.filename}</p>
                                  </div>
                                </div>
                              ) : (
                                <div className="relative">
                                  <img
                                    src={att.url}
                                    alt={att.filename}
                                    className="w-full h-32 object-cover rounded-lg cursor-pointer hover:opacity-90 transition-opacity"
                                    onLoad={() => handleImageLoad(att.url)}
                                    onError={() => handleImageError(att.url)}
                                  />
                                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all duration-200 rounded-lg flex items-center justify-center">
                                    <ZoomIn className="h-6 w-6 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                                  </div>
                                  <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-2 rounded-b-lg opacity-0 group-hover:opacity-100 transition-opacity">
                                    <p className="truncate">{att.filename}</p>
                                    <p className="text-gray-300">{new Date(att.uploadedDate).toLocaleDateString()}</p>
                                  </div>
                                </div>
                              )}
                            </>
                          ) : (
                            <div className="w-full h-32 bg-gray-100 rounded-lg flex items-center justify-center border border-gray-300 hover:bg-gray-200 transition-colors cursor-pointer">
                              <div className="text-center">
                                <FileText className="h-8 w-8 text-gray-500 mx-auto mb-2" />
                                <p className="text-xs text-gray-700 font-medium">{att.type.toUpperCase()}</p>
                                <p className="text-xs text-gray-500 truncate max-w-full px-2">{att.filename}</p>
                                <a
                                  href={att.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center mt-1 text-xs text-blue-600 hover:text-blue-800"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <ExternalLink className="h-3 w-3 mr-1" />
                                  Open
                                </a>
                              </div>
                            </div>
                          )}
                        </div>
                      ))
                    ) : (
                      <div className="col-span-2 sm:col-span-3 text-center py-8">
                        <ImageIcon className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                        <p className="text-sm text-gray-500">No attachments found for this report.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <div>
                <div className="card">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Admin Notes</h3>
                  <textarea
                    className="w-full p-2 border border-gray-300 rounded-md text-sm"
                    rows={4}
                    value={adminNotes}
                    onChange={(e) => setAdminNotes(e.target.value)}
                    placeholder="Add notes for this report..."
                  ></textarea>
                </div>
                <div className="card mt-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Actions</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => handleUpdateStatus('approved')}
                      disabled={updatingStatus}
                      className="w-full btn-primary flex items-center justify-center space-x-2"
                    >
                      <CheckCircle className="h-4 w-4" />
                      <span>{updatingStatus ? 'Approving...' : 'Approve'}</span>
                    </button>
                    <button
                      onClick={() => handleUpdateStatus('rejected')}
                      disabled={updatingStatus}
                      className="w-full btn-secondary flex items-center justify-center space-x-2 bg-red-500 hover:bg-red-600 text-white"
                    >
                      <X className="h-4 w-4" />
                      <span>{updatingStatus ? 'Rejecting...' : 'Reject'}</span>
                    </button>
                    <button
                      onClick={() => handleUpdateStatus('requires_action')}
                      disabled={updatingStatus}
                      className="w-full btn-secondary flex items-center justify-center space-x-2 bg-orange-500 hover:bg-orange-600 text-white"
                    >
                      <AlertTriangle className="h-4 w-4" />
                      <span>{updatingStatus ? 'Updating...' : 'Action Required'}</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Image Modal */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-95 flex items-center justify-center z-50">
          <div className="relative w-full h-full flex flex-col">
            {/* Header Controls */}
            <div className="absolute top-0 left-0 right-0 z-20 bg-gradient-to-b from-black/70 to-transparent p-4">
              <div className="flex items-center justify-between text-white">
                <div className="flex items-center space-x-4">
                  <h3 className="text-lg font-semibold">
                    {allImages[currentImageIndex]?.title || 'Image'}
                  </h3>
                  {allImages.length > 1 && (
                    <span className="text-sm text-gray-300">
                      {currentImageIndex + 1} of {allImages.length}
                    </span>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  {allImages.length > 1 && (
                    <button
                      onClick={openCollage}
                      className="bg-black/50 hover:bg-black/70 rounded-full p-2 transition-colors"
                      title="View all images"
                    >
                      <Grid3X3 className="h-5 w-5" />
                    </button>
                  )}
                  <a
                    href={selectedImage}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-black/50 hover:bg-black/70 rounded-full p-2 transition-colors"
                    title="Open original"
                  >
                    <ExternalLink className="h-5 w-5" />
                  </a>
                  <button
                    onClick={closeImageModal}
                    className="bg-black/50 hover:bg-black/70 rounded-full p-2 transition-colors"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Main Image Container */}
            <div className="flex-1 flex items-center justify-center p-4 pt-20 pb-20">
              {/* Previous Button */}
              {allImages.length > 1 && (
                <button
                  onClick={() => navigateImage('prev')}
                  className="absolute left-4 top-1/2 transform -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full p-3 transition-colors z-10"
                >
                  <ChevronLeft className="h-6 w-6" />
                </button>
              )}

              {/* Image */}
              <div className="relative max-w-full max-h-full">
                {selectedImage && imageLoadingStates[selectedImage] !== false && !imageErrorStates[selectedImage] && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
                  </div>
                )}

                {selectedImage && imageErrorStates[selectedImage] ? (
                  <div className="flex flex-col items-center justify-center bg-black/60 text-white px-6 py-8 rounded-lg" onClick={(e) => e.stopPropagation()}>
                    <ImageIcon className="h-12 w-12 mb-3 text-white/70" />
                    <p className="text-sm text-center">Unable to load image. Please try opening the original file.</p>
                  </div>
                ) : (
                  <img
                    src={selectedImage || ''}
                    alt={allImages[currentImageIndex]?.title || 'Full size view'}
                    className={`max-w-full max-h-full object-contain rounded-lg shadow-2xl`}
                    onLoad={() => selectedImage && handleImageLoad(selectedImage)}
                    onError={() => selectedImage && handleImageError(selectedImage)}
                    onClick={(e) => e.stopPropagation()}
                  />
                )}
              </div>

              {/* Next Button */}
              {allImages.length > 1 && (
                <button
                  onClick={() => navigateImage('next')}
                  className="absolute right-4 top-1/2 transform -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full p-3 transition-colors z-10"
                >
                  <ChevronRight className="h-6 w-6" />
                </button>
              )}
            </div>

            {/* Bottom Controls */}
            <div className="absolute bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-black/70 to-transparent p-4">
              <div className="flex items-center justify-between text-white">
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-gray-300">
                    Type: {allImages[currentImageIndex]?.type === 'answer' ? 'Answer Photo' : 'Attachment'}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  {allImages.length > 1 && (
                    <>
                      <button
                        onClick={() => navigateImage('prev')}
                        className="bg-black/50 hover:bg-black/70 rounded px-3 py-1 text-sm transition-colors"
                      >
                        Previous
                      </button>
                      <button
                        onClick={() => navigateImage('next')}
                        className="bg-black/50 hover:bg-black/70 rounded px-3 py-1 text-sm transition-colors"
                      >
                        Next
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Keyboard Navigation Hint */}
            {allImages.length > 1 && (
              <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 text-white/60 text-xs">
                Use ← → arrow keys to navigate
              </div>
            )}
          </div>

          {/* Click outside to close */}
          <div
            className="absolute inset-0 -z-10"
            onClick={closeImageModal}
          />
        </div>
      )}

      {/* Collage Modal */}
      {showCollage && allImages.length > 0 && (
        <div className="fixed inset-0 bg-black bg-opacity-95 flex items-center justify-center z-60">
          <div className="relative w-full h-full max-w-7xl mx-auto p-6">
            {/* Collage Header */}
            <div className="flex items-center justify-between text-white mb-6">
              <div>
                <h2 className="text-2xl font-bold">All Images</h2>
                <p className="text-gray-300">{allImages.length} images from this report</p>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setShowCollage(false)}
                  className="bg-black/50 hover:bg-black/70 rounded-full p-2 transition-colors"
                  title="Back to single view"
                >
                  <Maximize2 className="h-5 w-5" />
                </button>
                <button
                  onClick={closeImageModal}
                  className="bg-black/50 hover:bg-black/70 rounded-full p-2 transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Collage Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 max-h-[calc(100vh-120px)] overflow-y-auto">
              {allImages.map((image, index) => (
                <div
                  key={index}
                  className={`relative group cursor-pointer rounded-lg overflow-hidden transition-all duration-200 ${index === currentImageIndex
                    ? 'ring-4 ring-blue-500 ring-opacity-75 scale-105'
                    : 'hover:scale-105 hover:ring-2 hover:ring-white/50'
                    }`}
                  onClick={() => selectImageFromCollage(image.url)}
                >
                  <div className="aspect-square relative">
                    {imageLoadingStates[image.url] !== false && !imageErrorStates[image.url] && (
                      <div className="absolute inset-0 bg-gray-800 flex items-center justify-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                      </div>
                    )}
                    {imageErrorStates[image.url] ? (
                      <div className="w-full h-full bg-gray-800 flex items-center justify-center">
                        <div className="text-center text-white">
                          <ImageIcon className="h-8 w-8 mx-auto mb-2 opacity-50" />
                          <p className="text-xs opacity-75">Failed to load</p>
                        </div>
                      </div>
                    ) : (
                      <img
                        src={image.url}
                        alt={image.title}
                        className="w-full h-full object-cover"
                        onLoad={() => handleImageLoad(image.url)}
                        onError={() => handleImageError(image.url)}
                      />
                    )}

                    {/* Overlay */}
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all duration-200 flex items-center justify-center">
                      <ZoomIn className="h-6 w-6 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>

                    {/* Image Info */}
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <p className="text-white text-xs font-medium truncate">{image.title}</p>
                      <p className="text-gray-300 text-xs">
                        {image.type === 'answer' ? 'Answer Photo' : 'Attachment'}
                      </p>
                    </div>

                    {/* Current Image Indicator */}
                    {index === currentImageIndex && (
                      <div className="absolute top-2 right-2 bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold">
                        ✓
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Collage Footer */}
            <div className="flex items-center justify-center mt-6 text-white/60 text-sm">
              Click on any image to view it in full screen
            </div>
          </div>

          {/* Click outside to close */}
          <div
            className="absolute inset-0 -z-10"
            onClick={closeImageModal}
          />
        </div>
      )}
    </div>
  )
}
