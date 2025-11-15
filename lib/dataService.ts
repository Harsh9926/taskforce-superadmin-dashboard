import { collection, getDocs, getDoc, doc, updateDoc, deleteDoc, query, orderBy, limit, where, setDoc, onSnapshot, serverTimestamp, deleteField } from 'firebase/firestore';
import { db } from './firebase';

export interface DashboardStats {
  totalUsers: number;
  activeUsers: number;
  pendingRequests: number;
  totalComplaints: number;
  totalInspections: number;
  totalFeederPoints: number;
  totalTeams: number;
  totalIPRecords: number;
  // Role-based statistics
  adminUsers: number;
  taskForceUsers: number;
  commissionerUsers: number;
  inactiveUsers: number;
  // Additional role-based metrics
  activeAdmins: number;
  activeTaskForce: number;
  activeCommissioners: number;
}

export interface User {
  id: string;
  name: string;
  email: string;
  phone: string;
  role: string;
  organization?: string;
  department?: string;
  isActive: boolean;
  createdAt: any;
  lastLogin?: any;
}

export interface ComplianceReport {
  id: string;
  feederPointId: string;
  feederPointName: string;
  userId: string;
  userName: string;
  teamId: string;
  teamName: string;
  submittedAt: any; // Firestore timestamp
  submittedLocation: {
    latitude: number;
    longitude: number;
    address: string;
  };
  distanceFromFeederPoint: number; // in meters
  status: 'pending' | 'approved' | 'rejected' | 'requires_action';
  answers: ComplianceAnswer[];
  adminNotes?: string;
  reviewedBy?: string;
  reviewedAt?: any;
  createdAt: any;
  updatedAt: any;
  tripNumber: 1 | 2 | 3; // Which trip of the day (1st, 2nd, or 3rd)
  tripDate: string; // Date in YYYY-MM-DD format for daily tracking
  dailyTripId: string; // Unique identifier for the day's trips (userId_feederPointId_date)
  aiAnalysis?: string;
  ministryReport?: string;
  priority?: 'low' | 'medium' | 'high' | 'urgent';
  description?: string;
  title?: string;
  submittedBy?: string;
  attachments?: ComplianceReportAttachment[];
  rejectionAnalysis?: RejectionAnalysisResult;
  imageValidation?: ImageValidationResult;
}

export interface ComplianceAnswer {
  questionId: string;
  answer: 'yes' | 'no' | string;
  description?: string;
  photos?: string[]; // Array of photo URLs
  notes?: string;
}

export interface ComplianceReportAttachment {
  id: string;
  type: 'photo' | 'video' | 'audio' | 'document';
  url: string;
  filename: string;
  uploadedDate: string;
}

export interface RejectionAnalysisResult {
  reason: string;
  validationSummary: string;
  reviewedPhotos: string[];
  aiModel: string;
  generatedAt: string;
}

export interface ImageValidationResult {
  overallFindings: string;
  questionChecks: ImageValidationQuestionCheck[];
  aiModel: string;
  generatedAt: string;
}

export interface ImageValidationQuestionCheck {
  questionId: string;
  questionLabel?: string;
  answer?: string;
  status: 'match' | 'mismatch' | 'missing' | 'uncertain';
  summary: string;
  photoEvidence: string[];
}

export interface AccessRequest {
  id: string;
  name: string;
  email: string;
  phone: string;
  organization: string;
  department: string;
  requestedRole: string;
  status: 'pending' | 'approved' | 'rejected';
  createdAt: any;
  reviewedAt?: any;
  reviewedBy?: string;
}

export interface Complaint {
  id: string;
  title: string;
  description: string;
  category: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  status: 'open' | 'in_progress' | 'resolved' | 'closed';
  reportedBy: string;
  assignedTo?: string;
  createdAt: any;
  updatedAt: any;
}

export interface IPRecord {
  id: string;
  userId: string;
  userEmail: string;
  ipAddress: string;
  deviceInfo: string;
  location?: string;
  registeredAt: any;
  lastUsed: any;
  isActive: boolean;
}

export interface Team {
  id: string;
  name: string;
  members: User[];
}

export interface EmployeePerformance {
  userId: string;
  name: string;
  email: string;
  role: string;
  totalReports: number;
  approvedReports: number;
  rejectedReports: number;
  pendingReports: number;
  approvalRate: number;
  lastReportAt: Date | null;
}

export interface FeederPointSummary {
  key: string;
  feederPointId?: string;
  feederPointName: string;
  totalReports: number;
  approvedReports: number;
  rejectedReports: number;
  pendingReports: number;
  lastReportAt: Date | null;
  reports: ComplianceReport[];
}

export interface FeederPoint {
  id: string;
  name: string;
  assignedUserId?: string;
  assignedTeamId?: string;
  status: 'active' | 'maintenance' | 'inactive';
  location: {
    address: string;
    latitude: number;
    longitude: number;
  };
  priority: 'high' | 'medium' | 'low';
  lastInspection?: any; // Firestore timestamp
}

export class DataService {
  private static coerceDate(value: any): Date | null {
    if (!value) {
      return null;
    }
    if (value instanceof Date) {
      return value;
    }
    if (typeof value.toDate === 'function') {
      return value.toDate();
    }
    if (typeof value === 'string') {
      const parsed = new Date(value);
      return Number.isNaN(parsed.getTime()) ? null : parsed;
    }
    return null;
  }

  // Get dashboard statistics
  static async getDashboardStats(): Promise<DashboardStats> {
    try {
      const [
        usersSnapshot,
        activeUsersSnapshot,
        pendingRequestsSnapshot,
        complaintsSnapshot,
        inspectionsSnapshot,
        feederPointsSnapshot,
        teamsSnapshot,
        ipRecordsSnapshot
      ] = await Promise.all([
        getDocs(collection(db, 'approvedUsers')),
        getDocs(query(collection(db, 'approvedUsers'), where('isActive', '==', true))),
        getDocs(query(collection(db, 'accessRequests'), where('status', '==', 'pending'))),
        getDocs(collection(db, 'complianceReports')),
        getDocs(collection(db, 'inspections')),
        getDocs(collection(db, 'feederPoints')),
        getDocs(collection(db, 'teams')),
        getDocs(collection(db, 'ipRecords'))
      ]);

      // Calculate role-based statistics
      const allUsers = usersSnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as User));
      const activeUsers = activeUsersSnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as User));

      const adminUsers = allUsers.filter(user => user.role === 'admin').length;
      const taskForceUsers = allUsers.filter(user => user.role === 'task_force_team').length;
      const commissionerUsers = allUsers.filter(user => user.role === 'commissioner').length;
      const inactiveUsers = allUsers.filter(user => !user.isActive).length;

      const activeAdmins = activeUsers.filter(user => user.role === 'admin').length;
      const activeTaskForce = activeUsers.filter(user => user.role === 'task_force_team').length;
      const activeCommissioners = activeUsers.filter(user => user.role === 'commissioner').length;

      return {
        totalUsers: usersSnapshot.size,
        activeUsers: activeUsersSnapshot.size,
        pendingRequests: pendingRequestsSnapshot.size,
        totalComplaints: complaintsSnapshot.size,
        totalInspections: inspectionsSnapshot.size,
        totalFeederPoints: feederPointsSnapshot.size,
        totalTeams: teamsSnapshot.size,
        totalIPRecords: ipRecordsSnapshot.size,
        // Role-based statistics
        adminUsers,
        taskForceUsers,
        commissionerUsers,
        inactiveUsers,
        // Additional role-based metrics
        activeAdmins,
        activeTaskForce,
        activeCommissioners
      };
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
      return {
        totalUsers: 0,
        activeUsers: 0,
        pendingRequests: 0,
        totalComplaints: 0,
        totalInspections: 0,
        totalFeederPoints: 0,
        totalTeams: 0,
        totalIPRecords: 0,
        adminUsers: 0,
        taskForceUsers: 0,
        commissionerUsers: 0,
        inactiveUsers: 0,
        activeAdmins: 0,
        activeTaskForce: 0,
        activeCommissioners: 0
      };
    }
  }

  static async getEmployeePerformance(options?: {
    role?: string;
    startDate?: Date;
    endDate?: Date;
    includeInactive?: boolean;
  }): Promise<EmployeePerformance[]> {
    const roleFilter = options?.role;
    const startDate = options?.startDate ? new Date(options.startDate) : null;
    const endDate = options?.endDate ? new Date(options.endDate) : null;
    const includeInactive = options?.includeInactive ?? false;

    if (startDate && endDate && startDate > endDate) {
      return [];
    }

    try {
      const [usersSnapshot, reportsSnapshot] = await Promise.all([
        getDocs(collection(db, 'approvedUsers')),
        getDocs(collection(db, 'complianceReports'))
      ]);

      const users = usersSnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      } as User));

      const performanceByUser = new Map<string, EmployeePerformance>();
      const userEmailIndex = new Map<string, string>();
      const userNameIndex = new Map<string, string>();

      users.forEach(user => {
        if (roleFilter && user.role !== roleFilter) {
          return;
        }
        if (!includeInactive && user.isActive === false) {
          return;
        }
        performanceByUser.set(user.id, {
          userId: user.id,
          name: user.name || 'Unknown User',
          email: user.email || 'N/A',
          role: user.role || 'user',
          totalReports: 0,
          approvedReports: 0,
          rejectedReports: 0,
          pendingReports: 0,
          approvalRate: 0,
          lastReportAt: null
        });

        if (user.email) {
          userEmailIndex.set(user.email.toLowerCase(), user.id);
        }
        if (user.name) {
          userNameIndex.set(user.name.toLowerCase(), user.id);
        }
      });

      reportsSnapshot.docs.forEach(reportDoc => {
        const report = reportDoc.data() as ComplianceReport;
        const coalescedUserId = typeof report.userId === 'string' ? report.userId.trim() : report.userId || undefined;
        let stats = coalescedUserId ? performanceByUser.get(coalescedUserId) : undefined;

        if (!stats && report.submittedBy) {
          const emailKey = String(report.submittedBy).trim().toLowerCase();
          const emailMatch = userEmailIndex.get(emailKey);
          if (emailMatch) {
            stats = performanceByUser.get(emailMatch);
          }
        }

        if (!stats && report.userName) {
          const nameKey = String(report.userName).trim().toLowerCase();
          const nameMatch = userNameIndex.get(nameKey);
          if (nameMatch) {
            stats = performanceByUser.get(nameMatch);
          }
        }

        if (!stats) {
          return;
        }

        const reportDate =
          DataService.coerceDate(report.submittedAt) ||
          DataService.coerceDate(report.updatedAt) ||
          DataService.coerceDate(report.createdAt) ||
          DataService.coerceDate(report.tripDate);

        if (startDate && reportDate && reportDate < startDate) {
          return;
        }
        if (endDate && reportDate && reportDate > endDate) {
          return;
        }

        if (!reportDate && (startDate || endDate)) {
          return;
        }

        stats.totalReports += 1;

        if (report.status === 'approved') {
          stats.approvedReports += 1;
        } else if (report.status === 'rejected') {
          stats.rejectedReports += 1;
        } else {
          stats.pendingReports += 1;
        }

        if (reportDate && (!stats.lastReportAt || reportDate > stats.lastReportAt)) {
          stats.lastReportAt = reportDate;
        }
      });

      return Array.from(performanceByUser.values()).map(performance => ({
        ...performance,
        approvalRate: performance.totalReports
          ? performance.approvedReports / performance.totalReports
          : 0
      }));
    } catch (error) {
      console.error('Error fetching employee performance:', error);
      return [];
    }
  }

  static async getEmployeeReports(
    userId: string,
    options?: { startDate?: Date; endDate?: Date; userEmail?: string; userName?: string }
  ): Promise<ComplianceReport[]> {
    const startDate = options?.startDate ? new Date(options.startDate) : null;
    const endDate = options?.endDate ? new Date(options.endDate) : null;

    const reportsMap = new Map<string, ComplianceReport>();

    const collectReports = (snapshot: any) => {
      snapshot.docs.forEach((docSnapshot: any) => {
        const data = docSnapshot.data();
        const report = { ...data, id: docSnapshot.id } as ComplianceReport;

        const reportDate =
          DataService.coerceDate(report.submittedAt) ||
          DataService.coerceDate(report.updatedAt) ||
          DataService.coerceDate(report.createdAt) ||
          DataService.coerceDate(report.tripDate);

        if (startDate && reportDate && reportDate < startDate) {
          return;
        }
        if (endDate && reportDate && reportDate > endDate) {
          return;
        }
        if (!reportDate && (startDate || endDate)) {
          return;
        }

        reportsMap.set(report.id, report);
      });
    };

    if (userId) {
      const q = query(collection(db, 'complianceReports'), where('userId', '==', userId));
      const snapshot = await getDocs(q);
      collectReports(snapshot);
    }

    if (reportsMap.size === 0 && options?.userEmail) {
      const emailQuery = query(
        collection(db, 'complianceReports'),
        where('submittedBy', '==', options.userEmail)
      );
      const snapshot = await getDocs(emailQuery);
      collectReports(snapshot);
    }

    if (reportsMap.size === 0 && options?.userName) {
      const nameQuery = query(
        collection(db, 'complianceReports'),
        where('userName', '==', options.userName)
      );
      const snapshot = await getDocs(nameQuery);
      collectReports(snapshot);
    }

    if (reportsMap.size === 0) {
      const snapshot = await getDocs(collection(db, 'complianceReports'));
      snapshot.docs.forEach(docSnapshot => {
        const data = docSnapshot.data() as ComplianceReport;

        const matchesUser =
          data.userId === userId ||
          (options?.userEmail && data.submittedBy === options.userEmail) ||
          (options?.userName && data.userName === options.userName);

        if (!matchesUser) {
          return;
        }

        const report = { ...data, id: docSnapshot.id } as ComplianceReport;

        const reportDate =
          DataService.coerceDate(report.submittedAt) ||
          DataService.coerceDate(report.updatedAt) ||
          DataService.coerceDate(report.createdAt) ||
          DataService.coerceDate(report.tripDate);

        if (startDate && reportDate && reportDate < startDate) {
          return;
        }
        if (endDate && reportDate && reportDate > endDate) {
          return;
        }
        if (!reportDate && (startDate || endDate)) {
          return;
        }

        reportsMap.set(report.id, report);
      });
    }

    const reports = Array.from(reportsMap.values()).sort((a, b) => {
      const aDate =
        DataService.coerceDate(a.submittedAt) ||
        DataService.coerceDate(a.updatedAt) ||
        DataService.coerceDate(a.createdAt);
      const bDate =
        DataService.coerceDate(b.submittedAt) ||
        DataService.coerceDate(b.updatedAt) ||
        DataService.coerceDate(b.createdAt);

      const aTime = aDate ? aDate.getTime() : 0;
      const bTime = bDate ? bDate.getTime() : 0;
      return bTime - aTime;
    });

    return reports;
  }

  static async getFeederPointSummaries(options?: {
    startDate?: Date;
    endDate?: Date;
  }): Promise<FeederPointSummary[]> {
    const startDate = options?.startDate ? new Date(options.startDate) : null;
    const endDate = options?.endDate ? new Date(options.endDate) : null;

    const snapshot = await getDocs(collection(db, 'complianceReports'));

    const summaries = new Map<string, FeederPointSummary>();

    snapshot.docs.forEach(docSnapshot => {
      const data = docSnapshot.data() as ComplianceReport;
      const report = { ...data, id: docSnapshot.id } as ComplianceReport;

      const reportDate =
        DataService.coerceDate(report.submittedAt) ||
        DataService.coerceDate(report.updatedAt) ||
        DataService.coerceDate(report.createdAt) ||
        DataService.coerceDate(report.tripDate);

      if (startDate && reportDate && reportDate < startDate) {
        return;
      }
      if (endDate && reportDate && reportDate > endDate) {
        return;
      }
      if (!reportDate && (startDate || endDate)) {
        return;
      }

      const key = report.feederPointId || report.feederPointName || 'unspecified';
      if (!summaries.has(key)) {
        summaries.set(key, {
          key,
          feederPointId: report.feederPointId,
          feederPointName: report.feederPointName || 'Unspecified Feeder Point',
          totalReports: 0,
          approvedReports: 0,
          rejectedReports: 0,
          pendingReports: 0,
          lastReportAt: null,
          reports: []
        });
      }

      const summary = summaries.get(key)!;
      summary.totalReports += 1;
      summary.reports.push(report);

      if (report.status === 'approved') {
        summary.approvedReports += 1;
      } else if (report.status === 'rejected') {
        summary.rejectedReports += 1;
      } else {
        summary.pendingReports += 1;
      }

      if (reportDate && (!summary.lastReportAt || reportDate > summary.lastReportAt)) {
        summary.lastReportAt = reportDate;
      }
    });

    const results = Array.from(summaries.values()).map(summary => {
      summary.reports.sort((a, b) => {
        const aDate =
          DataService.coerceDate(a.submittedAt) ||
          DataService.coerceDate(a.updatedAt) ||
          DataService.coerceDate(a.createdAt);
        const bDate =
          DataService.coerceDate(b.submittedAt) ||
          DataService.coerceDate(b.updatedAt) ||
          DataService.coerceDate(b.createdAt);
        const aTime = aDate ? aDate.getTime() : 0;
        const bTime = bDate ? bDate.getTime() : 0;
        return bTime - aTime;
      });
      return summary;
    });

    return results.sort((a, b) => b.totalReports - a.totalReports);
  }

  static async getFeederPointReports(
    feederPointId?: string,
    options?: { fallbackName?: string; startDate?: Date; endDate?: Date }
  ): Promise<ComplianceReport[]> {
    const queries = [];

    if (feederPointId) {
      queries.push(
        getDocs(query(collection(db, 'complianceReports'), where('feederPointId', '==', feederPointId)))
      );
    }

    if (options?.fallbackName) {
      queries.push(
        getDocs(query(collection(db, 'complianceReports'), where('feederPointName', '==', options.fallbackName)))
      );
    }

    if (queries.length === 0) {
      return [];
    }

    const [startDate, endDate] = [
      options?.startDate ? new Date(options.startDate) : null,
      options?.endDate ? new Date(options.endDate) : null,
    ];

    const reportsMap = new Map<string, ComplianceReport>();

    const snapshots = await Promise.all(queries);
    snapshots.forEach(snapshot => {
      snapshot.docs.forEach(docSnapshot => {
        const data = docSnapshot.data() as ComplianceReport;
        const report = { ...data, id: docSnapshot.id } as ComplianceReport;

        const reportDate =
          DataService.coerceDate(report.submittedAt) ||
          DataService.coerceDate(report.updatedAt) ||
          DataService.coerceDate(report.createdAt) ||
          DataService.coerceDate(report.tripDate);

        if (startDate && reportDate && reportDate < startDate) {
          return;
        }
        if (endDate && reportDate && reportDate > endDate) {
          return;
        }
        if (!reportDate && (startDate || endDate)) {
          return;
        }

        reportsMap.set(report.id, report);
      });
    });

    const reports = Array.from(reportsMap.values());
    reports.sort((a, b) => {
      const aDate =
        DataService.coerceDate(a.submittedAt) ||
        DataService.coerceDate(a.updatedAt) ||
        DataService.coerceDate(a.createdAt);
      const bDate =
        DataService.coerceDate(b.submittedAt) ||
        DataService.coerceDate(b.updatedAt) ||
        DataService.coerceDate(b.createdAt);
      const aTime = aDate ? aDate.getTime() : 0;
      const bTime = bDate ? bDate.getTime() : 0;
      return bTime - aTime;
    });

    return reports;
  }

  // Get all users
  static onUsersChange(callback: (users: User[]) => void) {
    const q = query(collection(db, 'approvedUsers'), orderBy('createdAt', 'desc'));
    return onSnapshot(q, (snapshot) => {
      const users = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      } as User));
      callback(users);
    });
  }

  static async getAllUsers(): Promise<User[]> {
    const q = query(collection(db, 'approvedUsers'));
    const snapshot = await getDocs(q);
    const users = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    } as User));
    return users;
  }

  // Get all access requests
  static onAccessRequestsChange(callback: (requests: AccessRequest[]) => void) {
    const q = query(collection(db, 'accessRequests'), orderBy('createdAt', 'desc'));
    return onSnapshot(q, (snapshot) => {
      const requests = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      } as AccessRequest));
      callback(requests);
    });
  }

  // Get all complaints
  static onComplaintsChange(callback: (complaints: Complaint[]) => void) {
    const q = query(collection(db, 'complaints'), orderBy('createdAt', 'desc'));
    return onSnapshot(q, (snapshot) => {
      const complaints = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      } as Complaint));
      callback(complaints);
    });
  }

  static async getAllComplaints(): Promise<Complaint[]> {
    const q = query(collection(db, 'complaints'));
    const snapshot = await getDocs(q);
    const complaints = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    } as Complaint));
    return complaints;
  }

  // Get all compliance reports
  static async getAllComplianceReports(): Promise<ComplianceReport[]> {
    const q = query(collection(db, 'complianceReports'), orderBy('submittedAt', 'desc'));
    const snapshot = await getDocs(q);
    const reports = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    } as ComplianceReport));
    return reports;
  }

  static onComplianceReportsChange(callback: (reports: ComplianceReport[]) => void) {
    const q = query(collection(db, 'complianceReports'), orderBy('submittedAt', 'desc'));
    return onSnapshot(q, (snapshot) => {
      const reports = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      } as ComplianceReport));
      callback(reports);
    });
  }

  // Update compliance report status
  static async updateComplianceReportStatus(
    reportId: string,
    status: ComplianceReport['status'],
    adminNotes?: string,
    reviewedBy?: string,
    rejectionAnalysis?: RejectionAnalysisResult
  ): Promise<void> {
    try {
      const updateData: any = {
        status,
        reviewedAt: serverTimestamp(),
        updatedAt: serverTimestamp()
      };

      if (adminNotes) {
        updateData.adminNotes = adminNotes;
      }
      if (reviewedBy) {
        updateData.reviewedBy = reviewedBy;
      }
      if (status === 'rejected') {
        updateData.rejectionAnalysis = rejectionAnalysis ?? {
          reason: 'Automatic AI analysis could not be generated.',
          validationSummary: 'Gemini validation unavailable. Please add manual notes.',
          reviewedPhotos: [],
          aiModel: 'unavailable',
          generatedAt: new Date().toISOString()
        };
      } else {
        updateData.rejectionAnalysis = deleteField();
      }

      await updateDoc(doc(db, 'complianceReports', reportId), updateData);

      console.log('✅ Compliance report status updated:', reportId, status);
    } catch (error) {
      console.error('❌ Error updating compliance report status:', error);
      throw error;
    }
  }

  static async saveImageValidation(reportId: string, analysis: ImageValidationResult): Promise<void> {
    try {
      await updateDoc(doc(db, 'complianceReports', reportId), {
        imageValidation: analysis,
        updatedAt: serverTimestamp()
      });
    } catch (error) {
      console.error('Error saving image validation result:', error);
      throw error;
    }
  }

  // Get all IP records
  static onIPRecordsChange(callback: (records: IPRecord[]) => void) {
    const q = query(collection(db, 'ipRecords'), orderBy('registeredAt', 'desc'));
    return onSnapshot(q, (snapshot) => {
      const records = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      } as IPRecord));
      callback(records);
    });
  }

  // Get all feeder points
  static onFeederPointsChange(callback: (points: FeederPoint[]) => void) {
    const q = query(collection(db, 'feederPoints'));
    return onSnapshot(q, (snapshot) => {
      const points = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      } as FeederPoint));
      callback(points);
    });
  }

  static async getAllFeederPoints(): Promise<FeederPoint[]> {
    const q = query(collection(db, 'feederPoints'));
    const snapshot = await getDocs(q);
    const points = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    } as FeederPoint));
    return points;
  }

  // Get all teams
  static onTeamsChange(callback: (teams: Team[]) => void) {
    const q = query(collection(db, 'teams'));
    return onSnapshot(q, (snapshot) => {
      const teams = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      } as Team));
      callback(teams);
    });
  }

  static async updateFeederPoint(id: string, data: Partial<FeederPoint>): Promise<void> {
    const feederPointRef = doc(db, 'feederPoints', id);
    await updateDoc(feederPointRef, data);
  }

  static async createFeederPoint(data: Partial<FeederPoint>): Promise<void> {
    const feederPointsCollection = collection(db, 'feederPoints');
    await setDoc(doc(feederPointsCollection), data, { merge: true });
  }

  static async deleteFeederPoint(id: string): Promise<void> {
    const feederPointRef = doc(db, 'feederPoints', id);
    await deleteDoc(feederPointRef);
  }

  static async createSampleFeederPoints(): Promise<void> {
    const samplePoints = [
      { name: 'FP-001', status: 'active', priority: 'high', location: { address: '123 Main St', latitude: 34.0522, longitude: -118.2437 } },
      { name: 'FP-002', status: 'maintenance', priority: 'medium', location: { address: '456 Oak Ave', latitude: 34.0522, longitude: -118.2437 } },
      { name: 'FP-003', status: 'inactive', priority: 'low', location: { address: '789 Pine Ln', latitude: 34.0522, longitude: -118.2437 } },
    ];

    const feederPointsCollection = collection(db, 'feederPoints');
    for (const point of samplePoints) {
      await setDoc(doc(feederPointsCollection), point);
    }
  }

  static async getRecentActivity(): Promise<any[]> {
    try {
      const q = query(collection(db, 'recentActivity'), orderBy('timestamp', 'desc'), limit(10));
      const snapshot = await getDocs(q);
      const activity = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      return activity;
    } catch (error) {
      console.error('Error fetching recent activity:', error);
      return [];
    }
  }

  static async approveAccessRequest(request: AccessRequest): Promise<void> {
    try {
      const requestRef = doc(db, 'accessRequests', request.id);
      await updateDoc(requestRef, {
        status: 'approved',
        reviewedAt: serverTimestamp(),
        reviewedBy: 'AdminUserPlaceholder', // Replace with actual admin user ID/name
        updatedAt: serverTimestamp()
      });
      console.log('✅ Access request approved:', request.id);
    } catch (error) {
      console.error('❌ Error approving access request:', error);
      throw error;
    }
  }

  static async rejectAccessRequest(requestId: string): Promise<void> {
    try {
      const requestRef = doc(db, 'accessRequests', requestId);
      await updateDoc(requestRef, {
        status: 'rejected',
        reviewedAt: serverTimestamp(),
        reviewedBy: 'AdminUserPlaceholder', // Replace with actual admin user ID/name
        updatedAt: serverTimestamp()
      });
      console.log('✅ Access request rejected:', requestId);
    } catch (error) {
      console.error('❌ Error rejecting access request:', error);
      throw error;
    }
  }

  static async updateComplaint(complaintId: string, complaint: Complaint): Promise<void> {
    try {
      const complaintRef = doc(db, 'complaints', complaintId);
      await updateDoc(complaintRef, {
        ...complaint,
        updatedAt: serverTimestamp()
      });
      console.log('✅ Complaint updated:', complaintId);
    } catch (error) {
      console.error('❌ Error updating complaint:', error);
      throw error;
    }
  }

  static async deleteComplaint(complaintId: string): Promise<void> {
    try {
      await deleteDoc(doc(db, 'complaints', complaintId));
      console.log('✅ Complaint deleted:', complaintId);
    } catch (error) {
      console.error('❌ Error deleting complaint:', error);
      throw error;
    }
  }

  static async testDatabaseConnection(): Promise<boolean> {
    try {
      console.log("Attempting to fetch collections from Firestore...");
      const collections = await Promise.all([
        getDocs(collection(db, 'approvedUsers')),
        getDocs(collection(db, 'accessRequests')),
        getDocs(collection(db, 'complianceReports')),
        getDocs(collection(db, 'feederPoints')),
        getDocs(collection(db, 'teams')),
      ]);
      console.log('Successfully fetched collections:');
      console.log(`- approvedUsers: ${collections[0].size} documents`);
      console.log(`- accessRequests: ${collections[1].size} documents`);
      console.log(`- complianceReports: ${collections[2].size} documents`);
      console.log(`- feederPoints: ${collections[3].size} documents`);
      console.log(`- teams: ${collections[4].size} documents`);
      return true;
    } catch (error) {
      console.error("Error testing database connection:", error);
      return false;
    }
  }

  static async getUserReports(userId: string): Promise<ComplianceReport[]> {
    const q = query(collection(db, 'complianceReports'), where('userId', '==', userId));
    const snapshot = await getDocs(q);
    return snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as ComplianceReport));
  }

  static async getUserFeederPoints(userId: string): Promise<FeederPoint[]> {
    const q = query(collection(db, 'feederPoints'), where('assignedUserId', '==', userId));
    const snapshot = await getDocs(q);
    return snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as FeederPoint));
  }

  static async updateUser(id: string, data: Partial<User>): Promise<void> {
    const userRef = doc(db, 'approvedUsers', id);
    await updateDoc(userRef, data);
  }

  static async deleteUser(id: string): Promise<void> {
    const userRef = doc(db, 'approvedUsers', id);
    await deleteDoc(userRef);
  }
}
