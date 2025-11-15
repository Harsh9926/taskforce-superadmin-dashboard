import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  MapPin,
  Search,
  Filter,
  Eye,
  Edit,
  Trash2,
  Users,
  Calendar,
  AlertTriangle,
  CheckCircle,
  Clock,
  TrendingUp,
  Activity,
  Zap,
  Settings,
  User as UserIcon,
  Plus,
  FileText,
  RefreshCw,
  X,
  Camera
} from 'lucide-react'
import { ComplianceReport, DataService, FeederPoint, Team, User } from '@/lib/dataService'
import { AIService, ImageQuestionRule } from '@/lib/aiService'

export default function FeederPointsPage() {
  const [feederPoints, setFeederPoints] = useState<FeederPoint[]>([])
  const [teams, setTeams] = useState<Team[]>([])
  const [users, setUsers] = useState<User[]>([])
  const [enhancedFeederPoints, setEnhancedFeederPoints] = useState<any[]>([])
  const [filteredFeederPoints, setFilteredFeederPoints] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('active')
  const [assignmentFilter, setAssignmentFilter] = useState('all')
  const [selectedFeederPoint, setSelectedFeederPoint] = useState<any>(null)
  const [showDetailsModal, setShowDetailsModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [editingFeederPoint, setEditingFeederPoint] = useState<any | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error'>('connecting')
  const [creatingData, setCreatingData] = useState(false)
  const [feederPointReports, setFeederPointReports] = useState<ComplianceReport[]>([])
  const [reportsLoading, setReportsLoading] = useState(false)
  const [reportsError, setReportsError] = useState<string | null>(null)
  const [reportDateRange, setReportDateRange] = useState<{ start: string; end: string }>({ start: '', end: '' })
  const [reportStatusFilter, setReportStatusFilter] = useState<'all' | 'approved' | 'pending' | 'rejected' | 'requires_action'>('all')

  useEffect(() => {
    const unsubscribeFeederPoints = DataService.onFeederPointsChange(feederPointsData => {
      setFeederPoints(feederPointsData)
      setLoading(false)
    })

    const unsubscribeTeams = DataService.onTeamsChange(teamsData => {
      setTeams(teamsData)
    })

    const unsubscribeUsers = DataService.onUsersChange(usersData => {
      setUsers(usersData)
    })

    return () => {
      unsubscribeFeederPoints()
      unsubscribeTeams()
      unsubscribeUsers()
    }
  }, [])

  useEffect(() => {
    if (feederPoints.length > 0 && users.length > 0 && teams.length > 0) {
      const enhanced = feederPoints.map(fp => {
        let assignmentDetails = null

        if (fp.assignedUserId) {
          const assignedUser = users.find(user => user.id === fp.assignedUserId)
          if (assignedUser) {
            assignmentDetails = {
              type: 'individual',
              name: assignedUser.name || 'Unknown User',
              email: assignedUser.email || '',
              id: fp.assignedUserId,
              role: assignedUser.role || 'User'
            }
          }
        }

        if (fp.assignedTeamId) {
          const assignedTeam = teams.find(team => team.id === fp.assignedTeamId)
          if (assignedTeam) {
            const activeMembers = assignedTeam.members?.filter(member => member.isActive) || []
            assignmentDetails = {
              type: 'team',
              name: assignedTeam.name || 'Unknown Team',
              memberCount: activeMembers.length,
              id: fp.assignedTeamId,
              members: activeMembers
            }
          }
        }

        return {
          ...fp,
          assignmentDetails
        }
      })
      setEnhancedFeederPoints(enhanced)
    }
  }, [feederPoints, teams, users])

  useEffect(() => {
    filterFeederPoints()
  }, [feederPoints, searchTerm, statusFilter, assignmentFilter])

  const loadFeederPoints = async () => {
    setLoading(true);
    const points = await DataService.getAllFeederPoints();
    setFeederPoints(points);
    setLoading(false);
  };

  const filterFeederPoints = () => {
    let filtered = enhancedFeederPoints

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(fp =>
        fp.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        fp.location?.address?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        fp.assignmentDetails?.name?.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    // Status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(fp => fp.status === statusFilter)
    }

    // Assignment filter
    if (assignmentFilter !== 'all') {
      if (assignmentFilter === 'assigned') {
        filtered = filtered.filter(fp => fp.assignmentDetails)
      } else if (assignmentFilter === 'unassigned') {
        filtered = filtered.filter(fp => !fp.assignmentDetails)
      } else if (assignmentFilter === 'individual') {
        filtered = filtered.filter(fp => fp.assignmentDetails?.type === 'individual')
      } else if (assignmentFilter === 'team') {
        filtered = filtered.filter(fp => fp.assignmentDetails?.type === 'team')
      }
    }

    setFilteredFeederPoints(filtered)
  }

  const handleReportUpdated = useCallback((reportId?: string, changes: Partial<ComplianceReport> = {}) => {
    if (!reportId) return
    setFeederPointReports(prev =>
      prev.map(report => (report.id === reportId ? { ...report, ...changes } : report))
    )
  }, [])

  const loadReportsForFeederPoint = async (
    feederPoint: FeederPoint | null,
    customRange?: { start: string; end: string }
  ) => {
    if (!feederPoint) return

    setReportsLoading(true)
    setReportsError(null)
    setFeederPointReports([])

    try {
      const range = customRange || reportDateRange
      const startDate = range.start ? new Date(range.start) : undefined
      const endDate = range.end ? new Date(range.end) : undefined

      const reports = await DataService.getFeederPointReports(feederPoint.id, {
        fallbackName: feederPoint.name,
        startDate,
        endDate
      })
      setFeederPointReports(reports)
    } catch (error) {
      console.error('Error fetching feeder point reports:', error)
      setReportsError('Unable to load compliance reports. Please try again.')
    } finally {
      setReportsLoading(false)
    }
  }

  const handleViewDetails = async (feederPoint: any) => {
    const resetRange = { start: '', end: '' }
    setReportDateRange(resetRange)
    setReportStatusFilter('all')
    setSelectedFeederPoint(feederPoint)
    setShowDetailsModal(true)
    await loadReportsForFeederPoint(feederPoint, resetRange)
  }

  const handleEditFeederPoint = (feederPoint: any) => {
    setEditingFeederPoint({ ...feederPoint })
    setShowEditModal(true)
  }

  const handleSaveFeederPoint = async () => {
    if (!editingFeederPoint) return

    try {
      await DataService.updateFeederPoint(editingFeederPoint.id, editingFeederPoint)
      await loadFeederPoints()
      setShowEditModal(false)
      setEditingFeederPoint(null)
      alert('Feeder point updated successfully!')
    } catch (error) {
      console.error('Error updating feeder point:', error)
      alert('Error updating feeder point. Please try again.')
    }
  }

  const handleCreateFeederPoint = async (feederPointData: any) => {
    try {
      await DataService.createFeederPoint(feederPointData)
      await loadFeederPoints()
      setShowCreateModal(false)
      alert('Feeder point created successfully!')
    } catch (error) {
      console.error('Error creating feeder point:', error)
      alert('Error creating feeder point. Please try again.')
    }
  }

  const handleDeleteFeederPoint = async (feederPoint: any) => {
    if (!confirm(`Are you sure you want to delete feeder point "${feederPoint.name}"?`)) {
      return
    }

    try {
      await DataService.deleteFeederPoint(feederPoint.id)
      await loadFeederPoints()
      alert('Feeder point deleted successfully!')
    } catch (error) {
      console.error('Error deleting feeder point:', error)
      alert('Error deleting feeder point. Please try again.')
    }
  }

  const createSampleData = async () => {
    setCreatingData(true)
    try {
      await DataService.createSampleFeederPoints()
      alert('Sample feeder points created successfully!')
      await loadFeederPoints() // Reload data
    } catch (error) {
      console.error('Error creating sample data:', error)
      alert('Error creating sample data. Check console for details.')
    } finally {
      setCreatingData(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800'
      case 'maintenance':
        return 'bg-yellow-100 text-yellow-800'
      case 'inactive':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-100 text-red-800'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800'
      case 'low':
        return 'bg-green-100 text-green-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  const stats = {
    total: enhancedFeederPoints.length,
    active: enhancedFeederPoints.filter(fp => fp.status === 'active').length,
    maintenance: enhancedFeederPoints.filter(fp => fp.status === 'maintenance').length,
    inactive: enhancedFeederPoints.filter(fp => fp.status === 'inactive').length,
    assigned: enhancedFeederPoints.filter(fp => fp.assignmentDetails).length,
    unassigned: enhancedFeederPoints.filter(fp => !fp.assignmentDetails).length,
    individual: enhancedFeederPoints.filter(fp => fp.assignmentDetails?.type === 'individual').length,
    team: enhancedFeederPoints.filter(fp => fp.assignmentDetails?.type === 'team').length
  }

  return (
    <div className="space-y-6 scrollable-content">
      {/* Page Header */}
      <div className="border-b border-gray-200 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Zap className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">Feeder Points Monitor</h1>
              <p className="mt-1 text-sm sm:text-base text-gray-600">
                Real-time monitoring of all feeder points and assignments
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowCreateModal(true)}
              className="btn-primary flex items-center space-x-2"
            >
              <Plus className="h-4 w-4" />
              <span>Create Feeder Point</span>
            </button>
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
              connectionStatus === 'connected' ? 'bg-green-100 text-green-800' :
              connectionStatus === 'error' ? 'bg-red-100 text-red-800' :
              'bg-yellow-100 text-yellow-800'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-500' :
                connectionStatus === 'error' ? 'bg-red-500' :
                'bg-yellow-500'
              }`} />
              <span>
                {connectionStatus === 'connected' ? 'Connected' :
                 connectionStatus === 'error' ? 'Connection Error' :
                 'Connecting...'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <div className="stat-card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-blue-500">
              <Zap className="h-6 w-6 text-white" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Points</p>
              <p className="text-2xl font-semibold text-gray-900">{stats.total}</p>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-green-500">
              <CheckCircle className="h-6 w-6 text-white" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Active</p>
              <p className="text-2xl font-semibold text-gray-900">{stats.active}</p>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-yellow-500">
              <Settings className="h-6 w-6 text-white" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Maintenance</p>
              <p className="text-2xl font-semibold text-gray-900">{stats.maintenance}</p>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center">
            <div className="p-3 rounded-lg bg-purple-500">
              <Users className="h-6 w-6 text-white" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Assigned</p>
              <p className="text-2xl font-semibold text-gray-900">{stats.assigned}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Assignment Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Assignment Status</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Assigned</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full" 
                    style={{ width: `${(stats.assigned / stats.total) * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-semibold text-gray-900">{stats.assigned}</span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Unassigned</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-red-500 h-2 rounded-full" 
                    style={{ width: `${(stats.unassigned / stats.total) * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-semibold text-gray-900">{stats.unassigned}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Assignment Types</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Individual</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full" 
                    style={{ width: `${stats.assigned > 0 ? (stats.individual / stats.assigned) * 100 : 0}%` }}
                  ></div>
                </div>
                <span className="text-sm font-semibold text-gray-900">{stats.individual}</span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Team</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-purple-500 h-2 rounded-full" 
                    style={{ width: `${stats.assigned > 0 ? (stats.team / stats.assigned) * 100 : 0}%` }}
                  ></div>
                </div>
                <span className="text-sm font-semibold text-gray-900">{stats.team}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="card">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0 sm:space-x-4">
          <div className="flex-1 max-w-md">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
              <input
                type="text"
                placeholder="Search feeder points..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="flex space-x-4">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="active">Active</option>
              <option value="maintenance">Maintenance</option>
              <option value="inactive">Inactive</option>
            </select>

            <select
              value={assignmentFilter}
              onChange={(e) => setAssignmentFilter(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="all">All Assignments</option>
              <option value="assigned">Assigned</option>
              <option value="unassigned">Unassigned</option>
              <option value="individual">Individual</option>
              <option value="team">Team</option>
            </select>
          </div>
        </div>
      </div>

      {/* Feeder Points Table */}
      <div className="card">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Feeder Points ({filteredFeederPoints.length})
            </h2>
            <div className="flex items-center space-x-2 text-sm text-gray-500">
              <Activity className="h-4 w-4" />
              <span>Real-time data</span>
            </div>
          </div>

          <div className="scrollable-table">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Feeder Point
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Location
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Assignment
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Priority
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Last Inspection
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredFeederPoints.map((feederPoint, index) => (
                  <tr key={feederPoint.id || index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 h-10 w-10">
                          <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                            <Zap className="h-5 w-5 text-blue-600" />
                          </div>
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">
                            {feederPoint.name || `Feeder Point ${index + 1}`}
                          </div>
                          <div className="text-sm text-gray-500">
                            ID: {feederPoint.id}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <MapPin className="h-4 w-4 text-gray-400 mr-2" />
                        <div>
                          <div className="text-sm text-gray-900">
                            {feederPoint.location?.address ||
                             feederPoint.address ||
                             (feederPoint.location?.latitude && feederPoint.location?.longitude
                               ? `${feederPoint.location.latitude}, ${feederPoint.location.longitude}`
                               : 'No location data')}
                          </div>
                          {feederPoint.location?.latitude && feederPoint.location?.longitude && (
                            <div className="text-xs text-gray-500">
                              {feederPoint.location.latitude.toFixed(4)}, {feederPoint.location.longitude.toFixed(4)}
                            </div>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(feederPoint.status)}`}>
                        {feederPoint.status || 'unknown'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {feederPoint.assignmentDetails ? (
                        <div className="flex items-center">
                          {feederPoint.assignmentDetails.type === 'individual' ? (
                            <UserIcon className="h-4 w-4 text-blue-500 mr-2" />
                          ) : (
                            <Users className="h-4 w-4 text-purple-500 mr-2" />
                          )}
                          <div>
                            <div className="text-sm font-medium text-gray-900">
                              {feederPoint.assignmentDetails.name}
                            </div>
                            <div className="text-xs text-gray-500">
                              {feederPoint.assignmentDetails.type === 'individual' ? 'Individual' : `Team (${feederPoint.assignmentDetails.memberCount} members)`}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <span className="text-sm text-gray-500">Unassigned</span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getPriorityColor(feederPoint.priority)}`}>
                        {feederPoint.priority || 'normal'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {feederPoint.lastInspection ? (
                        <div className="flex items-center">
                          <Calendar className="h-4 w-4 mr-1" />
                          {feederPoint.lastInspection.toDate?.()?.toLocaleDateString() || 'Invalid date'}
                        </div>
                      ) : (
                        <span className="text-gray-400">Never</span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button
                        onClick={() => handleViewDetails(feederPoint)}
                        className="text-primary-600 hover:text-primary-900 mr-3"
                        title="View Details"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => handleEditFeederPoint(feederPoint)}
                        className="text-yellow-600 hover:text-yellow-900 mr-3"
                        title="Edit Feeder Point"
                      >
                        <Edit className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => handleDeleteFeederPoint(feederPoint)}
                        className="text-red-600 hover:text-red-900"
                        title="Delete Feeder Point"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {filteredFeederPoints.length === 0 && !loading && (
            <div className="text-center py-12">
              <Zap className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">
                {feederPoints.length === 0 ? 'No feeder points in database' : 'No feeder points match your filters'}
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                {feederPoints.length === 0
                  ? 'Feeder points will appear here when they are created in the mobile app.'
                  : searchTerm || statusFilter !== 'all' || assignmentFilter !== 'all'
                  ? 'Try adjusting your search or filters.'
                  : 'All feeder points are currently filtered out.'}
              </p>
              {feederPoints.length === 0 && (
                <div className="mt-4 space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-800">
                      <strong>Debug Info:</strong> Connected to database but no feeder points found.
                      Check the browser console for more details.
                    </p>
                  </div>
                  <button
                    onClick={createSampleData}
                    disabled={creatingData}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                  >
                    {creatingData ? 'Creating Sample Data...' : 'Create Sample Feeder Points'}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Feeder Point Details Modal */}
      {showDetailsModal && selectedFeederPoint && (
        <FeederPointDetailsModal
          feederPoint={selectedFeederPoint}
          reports={feederPointReports}
          loading={reportsLoading}
          error={reportsError}
          dateRange={reportDateRange}
          onDateRangeChange={range => setReportDateRange(range)}
          onApplyDateRange={range => {
            if (selectedFeederPoint) {
              loadReportsForFeederPoint(selectedFeederPoint, range)
            }
          }}
          statusFilter={reportStatusFilter}
          onStatusFilterChange={value => setReportStatusFilter(value)}
          onReportUpdated={handleReportUpdated}
          onClose={() => {
            setShowDetailsModal(false)
            setSelectedFeederPoint(null)
            setFeederPointReports([])
            setReportsError(null)
            setReportDateRange({ start: '', end: '' })
            setReportStatusFilter('all')
          }}
          onRefresh={() => loadReportsForFeederPoint(selectedFeederPoint)}
        />
      )}

      {/* Create Feeder Point Modal */}
      {showCreateModal && (
        <FeederPointFormModal
          title="Create Feeder Point"
          onClose={() => setShowCreateModal(false)}
          onSave={handleCreateFeederPoint}
        />
      )}

      {/* Edit Feeder Point Modal */}
      {showEditModal && editingFeederPoint && (
        <FeederPointFormModal
          title="Edit Feeder Point"
          feederPoint={editingFeederPoint}
          onClose={() => setShowEditModal(false)}
          onSave={handleSaveFeederPoint}
        />
      )}
    </div>
  )
}

interface FeederPointDetailsModalProps {
  feederPoint: any
  reports: ComplianceReport[]
  loading: boolean
  error: string | null
  dateRange: { start: string; end: string }
  onDateRangeChange: (range: { start: string; end: string }) => void
  onApplyDateRange: (range: { start: string; end: string }) => void
  statusFilter: 'all' | 'approved' | 'pending' | 'rejected' | 'requires_action'
  onStatusFilterChange: (value: 'all' | 'approved' | 'pending' | 'rejected' | 'requires_action') => void
  onReportUpdated: (reportId?: string, changes?: Partial<ComplianceReport>) => void
  onClose: () => void
  onRefresh: () => void
}

function FeederPointDetailsModal({
  feederPoint,
  reports,
  loading,
  error,
  dateRange,
  onDateRangeChange,
  onApplyDateRange,
  statusFilter,
  onStatusFilterChange,
  onReportUpdated,
  onClose,
  onRefresh
}: FeederPointDetailsModalProps) {
  const summary = useMemo(
    () => createFeederPointReportSummary(feederPoint, reports),
    [feederPoint, reports]
  )
  const filteredReports = useMemo(() => {
    if (statusFilter === 'all') return reports
    return reports.filter(report => (report.status || 'pending') === statusFilter)
  }, [reports, statusFilter])
  const statusOptions: { label: string; value: FeederPointDetailsModalProps['statusFilter'] }[] = [
    { label: 'All', value: 'all' },
    { label: 'Approved', value: 'approved' },
    { label: 'Pending', value: 'pending' },
    { label: 'Rejected', value: 'rejected' },
    { label: 'Requires Action', value: 'requires_action' }
  ]
  const [selectedReport, setSelectedReport] = useState<ComplianceReport | null>(null)
  const [imageAnalysisLoading, setImageAnalysisLoading] = useState(false)
  const [imageAnalysisError, setImageAnalysisError] = useState<string | null>(null)

  useEffect(() => {
    if (!selectedReport) return
    const exists = filteredReports.some(report =>
      (report.id || report.dailyTripId) === (selectedReport.id || selectedReport.dailyTripId)
    )
    if (!exists) {
      setSelectedReport(null)
    }
  }, [filteredReports, selectedReport])

  useEffect(() => {
    if (!selectedReport) return
    if (selectedReport.imageValidation) {
      setImageAnalysisError(null)
      return
    }
    if (!reportHasPhotos(selectedReport)) {
      setImageAnalysisError('No photos attached for ChatGPT to validate for this report.')
      return
    }
    let cancelled = false
    const runAnalysis = async () => {
      setImageAnalysisLoading(true)
      setImageAnalysisError(null)
      try {
        const validation = await AIService.analyzeReportImages(selectedReport, IMAGE_VALIDATION_RULES)
        if (cancelled) return
        if (selectedReport.id) {
          await DataService.saveImageValidation(selectedReport.id, validation)
          onReportUpdated(selectedReport.id, { imageValidation: validation })
        }
        setSelectedReport(prev =>
          prev && prev.id === selectedReport.id ? { ...prev, imageValidation: validation } : prev
        )
      } catch (error) {
        console.error('Error running ChatGPT image validation:', error)
        if (!cancelled) {
          setImageAnalysisError('Unable to analyze photos right now. Please try again later.')
        }
      } finally {
        if (!cancelled) {
          setImageAnalysisLoading(false)
        }
      }
    }
    runAnalysis()
    return () => {
      cancelled = true
    }
  }, [selectedReport, onReportUpdated])

  const handleRangeChange = (field: 'start' | 'end', value: string) => {
    onDateRangeChange({
      ...dateRange,
      [field]: value
    })
  }

  const handleApplyRange = () => {
    onApplyDateRange(dateRange)
  }

  const handleClearRange = () => {
    const cleared = { start: '', end: '' }
    onDateRangeChange(cleared)
    onApplyDateRange(cleared)
  }

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center px-4 py-6">
        <div className="fixed inset-0 bg-gray-900/50" onClick={onClose} />
        <div className="relative z-10 w-full max-w-6xl overflow-hidden rounded-2xl bg-white shadow-xl">
          <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4">
            <div className="flex items-center space-x-3">
              <FileText className="h-6 w-6 text-blue-600" />
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Feeder Point Compliance Summary</h2>
                <p className="text-sm text-gray-500">{feederPoint?.name || 'Feeder Point Details'}</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={onRefresh}
                disabled={loading}
                className="inline-flex items-center rounded-full border border-gray-200 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
              >
                <RefreshCw className={`mr-1.5 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh Data
              </button>
              <button
                onClick={onClose}
                className="rounded-full border border-gray-200 p-2 text-gray-500 hover:bg-gray-50"
                aria-label="Close report modal"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="border-b border-gray-100 bg-gray-50 px-6 py-4">
            <div className="flex flex-wrap items-end gap-4">
              <div className="flex flex-col text-sm text-gray-600">
                <label className="mb-1 font-medium text-gray-700">From</label>
                <input
                  type="date"
                  value={dateRange.start}
                  onChange={e => handleRangeChange('start', e.target.value)}
                  className="rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div className="flex flex-col text-sm text-gray-600">
                <label className="mb-1 font-medium text-gray-700">To</label>
                <input
                  type="date"
                  value={dateRange.end}
                  onChange={e => handleRangeChange('end', e.target.value)}
                  className="rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-900 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={handleApplyRange}
                  disabled={loading}
                  className="inline-flex items-center rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                >
                  Apply Range
                </button>
                <button
                  onClick={handleClearRange}
                  disabled={loading || (!dateRange.start && !dateRange.end)}
                  className="inline-flex items-center rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-white disabled:opacity-50"
                >
                  Clear
                </button>
              </div>
            </div>
          </div>

          <div className="max-h-[80vh] overflow-y-auto px-6 py-6">
            {loading ? (
              <div className="flex h-72 flex-col items-center justify-center space-y-3">
                <div className="h-12 w-12 animate-spin rounded-full border-b-2 border-blue-600" />
                <p className="text-sm text-gray-500">Loading compliance reports…</p>
              </div>
            ) : (
              <div className="space-y-6">
                {error && (
                  <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                    {error}
                  </div>
                )}

                <section>
                  <h3 className="text-xl font-semibold text-gray-900">1. Basic Information</h3>
                  <div className="mt-3 grid grid-cols-1 gap-4 rounded-xl border border-gray-100 bg-gray-50 p-4 text-sm text-gray-700 sm:grid-cols-2">
                    <p><span className="font-semibold">Feeder Point Name:</span> {summary.basicInfo.name}</p>
                    <p><span className="font-semibold">Zone:</span> {summary.basicInfo.zone}</p>
                    <p><span className="font-semibold">Ward:</span> {summary.basicInfo.ward}</p>
                    <p><span className="font-semibold">Date:</span> {summary.basicInfo.date}</p>
                    <p><span className="font-semibold">Total Trips:</span> {summary.basicInfo.totalTrips}</p>
                    <p><span className="font-semibold">Submitted By:</span> {summary.basicInfo.submittedBy}</p>
                  </div>
                </section>

                <section>
                  <h3 className="text-xl font-semibold text-gray-900">2. Overall Status (Bullet Format)</h3>
                  <ul className="mt-3 space-y-1 rounded-xl border border-gray-100 bg-white p-4 text-sm text-gray-700">
                    <li>Feeder Point Cleanliness: {summary.overallStatus.cleanliness}</li>
                    <li>Segregation Status: {summary.overallStatus.segregation}</li>
                    <li>Vehicle Availability: {summary.overallStatus.vehicle}</li>
                    <li>Swach Worker Presence: {summary.overallStatus.swachWorkers}</li>
                    <li>Nearby Area Cleanliness: {summary.overallStatus.nearbyCleanliness}</li>
                    <li>Signboard/QR Availability: {summary.overallStatus.signboard}</li>
                    <li>Final Compliance Score: {summary.overallStatus.finalScore}</li>
                  </ul>
                </section>

                <section>
                  <h3 className="text-xl font-semibold text-gray-900">3. Trip-wise Summary (Table Format)</h3>
                  <div className="mt-3 overflow-hidden rounded-xl border border-gray-100">
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200 text-sm">
                        <thead className="bg-gray-50 text-gray-600">
                          <tr>
                            <th className="px-4 py-3 text-left font-semibold">Trip</th>
                            <th className="px-4 py-3 text-left font-semibold">Time</th>
                            <th className="px-4 py-3 text-left font-semibold">Distance</th>
                            <th className="px-4 py-3 text-left font-semibold">Clean</th>
                            <th className="px-4 py-3 text-left font-semibold">Segregated</th>
                            <th className="px-4 py-3 text-left font-semibold">Vehicle</th>
                            <th className="px-4 py-3 text-left font-semibold">Workers</th>
                            <th className="px-4 py-3 text-left font-semibold">Photo Attached</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100 bg-white text-gray-800">
                          {summary.tripSummaries.map(row => (
                            <tr key={row.tripLabel}>
                              <td className="whitespace-nowrap px-4 py-3 font-medium">{row.tripLabel}</td>
                              <td className="whitespace-nowrap px-4 py-3">{row.time}</td>
                              <td className="whitespace-nowrap px-4 py-3">{row.distance}</td>
                              <td className="whitespace-nowrap px-4 py-3">{row.clean}</td>
                              <td className="whitespace-nowrap px-4 py-3">{row.segregated}</td>
                              <td className="whitespace-nowrap px-4 py-3">{row.vehicle}</td>
                              <td className="whitespace-nowrap px-4 py-3">{row.workers}</td>
                              <td className="whitespace-nowrap px-4 py-3">{row.photoAttached}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </section>

                <section>
                  <h3 className="text-xl font-semibold text-gray-900">4. Key Observations (4–5 Bullet Points)</h3>
                  <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-gray-700">
                    {summary.observations.map((item, index) => (
                      <li key={`observation-${index}`}>{item}</li>
                    ))}
                  </ul>
                </section>

                <section>
                  <h3 className="text-xl font-semibold text-gray-900">5. Photo Evidence Section</h3>
                  <div className="mt-3 grid gap-4 md:grid-cols-3">
                    {summary.photoEvidence.map((item, index) => (
                      <div key={item.label} className="flex flex-col rounded-xl border border-gray-100 bg-white p-4 text-sm text-gray-700">
                        <div className="mb-2 flex items-center justify-between">
                          <span className="font-semibold">{item.label}</span>
                          <Camera className="h-4 w-4 text-gray-400" />
                        </div>
                        {item.photoUrl ? (
                          <div className="h-32 overflow-hidden rounded-lg border border-gray-100">
                            <a
                              href={item.photoUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block h-full w-full transition-transform duration-200 hover:scale-105"
                              title="Open full-size photo"
                            >
                              <img
                                src={item.photoUrl}
                                alt={item.label}
                                className="h-full w-full object-cover"
                              />
                            </a>
                          </div>
                        ) : (
                          <div className="flex h-32 items-center justify-center rounded-lg border border-dashed border-gray-200 text-xs text-gray-400">
                            No photo uploaded
                          </div>
                        )}
                        <span className="mt-2 text-xs text-gray-500">{summary.tripSummaries[index]?.time || '—'}</span>
                      </div>
                    ))}
                  </div>
                </section>

                <section>
                  <h3 className="text-xl font-semibold text-gray-900">Report Library</h3>
                  <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
                    <p className="text-sm text-gray-600">Complete list of compliance reports captured for this feeder point.</p>
                    <div className="flex flex-wrap gap-2 text-xs font-medium">
                      {statusOptions.map(option => (
                        <button
                          key={option.value}
                          onClick={() => onStatusFilterChange(option.value)}
                          className={`rounded-full px-3 py-1 ${
                            statusFilter === option.value
                              ? 'bg-blue-600 text-white'
                              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                          }`}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="mt-3 max-h-64 space-y-3 overflow-y-auto rounded-xl border border-gray-100 bg-white p-3">
                    {reports.length === 0 && (
                      <div className="text-sm text-gray-500">
                        No compliance reports are available for this feeder point yet.
                      </div>
                    )}
                    {reports.length > 0 && filteredReports.length === 0 && (
                      <div className="text-sm text-gray-500">
                        No reports match the selected status filter.
                      </div>
                    )}
                    {filteredReports.map(report => {
                      const displayTitle = report.title?.trim() ? report.title : 'Untitled Report'
                      const displayDate = formatReportDate(report) || 'Unscheduled Submission'
                      return (
                        <div key={report.id || report.dailyTripId} className="rounded-lg border border-gray-100 p-3">
                          <div className="flex flex-wrap items-center justify-between text-sm text-gray-600">
                            <span className="font-semibold text-gray-900">{displayTitle}</span>
                            <span className="rounded-full bg-gray-50 px-2 py-0.5 text-xs capitalize text-gray-700">
                              {report.status || 'pending'}
                            </span>
                          </div>
                          <p className="mt-0.5 text-xs text-gray-500">{displayDate}</p>
                        {report.description && (
                          <p className="text-xs text-gray-500">{report.description}</p>
                        )}
                        <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-gray-500">
                          <span>Trip {report.tripNumber || '–'}</span>
                          <span>{report.userName || report.submittedBy || 'Unknown submitter'}</span>
                          <span>{report.attachments?.length || 0} attachment(s)</span>
                        </div>
                        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                          <span className="text-xs text-gray-500">
                            {report.priority ? `Priority: ${report.priority}` : ''}
                          </span>
                          <button
                            onClick={() => setSelectedReport(report)}
                            className="inline-flex items-center rounded-full border border-blue-100 p-2 text-blue-600 hover:bg-blue-50"
                            aria-label="View report details"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                        </div>
                        </div>
                      )
                    })}
                  </div>
                </section>

                {selectedReport && (
                  <section>
                    <div className="flex items-center justify-between">
                      <h3 className="text-xl font-semibold text-gray-900">Report Details</h3>
                      <button
                        onClick={() => setSelectedReport(null)}
                        className="text-sm text-gray-500 hover:text-gray-700"
                      >
                        Close
                      </button>
                    </div>
                    <div className="mt-3 space-y-4 rounded-xl border border-gray-100 bg-white p-4 text-sm text-gray-700">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div>
                          <p className="text-lg font-semibold text-gray-900">{selectedReport.title || 'Untitled Report'}</p>
                          <p className="text-xs text-gray-500">{formatReportDate(selectedReport)}</p>
                        </div>
                        <span className={`rounded-full px-3 py-1 text-xs font-semibold capitalize ${getStatusBadgeClasses(selectedReport.status)}`}>
                          {selectedReport.status || 'pending'}
                        </span>
                      </div>
                      {selectedReport.description && (
                        <p className="text-sm text-gray-600">{selectedReport.description}</p>
                      )}
                      <div className="grid gap-4 md:grid-cols-2">
                        <DetailField label="Submitted By" value={selectedReport.userName || selectedReport.submittedBy || 'Unknown'} />
                        <DetailField label="Trip Number" value={selectedReport.tripNumber ? `Trip ${selectedReport.tripNumber}` : 'Not specified'} />
                        <DetailField label="Distance" value={typeof selectedReport.distanceFromFeederPoint === 'number' ? formatDistance(selectedReport.distanceFromFeederPoint) : 'Not recorded'} />
                        <DetailField label="Location" value={selectedReport.submittedLocation?.address || 'Not recorded'} />
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-gray-900">Question Responses</h4>
                        {selectedReport.answers && selectedReport.answers.length > 0 ? (
                          <div className="mt-2 space-y-2">
                            {selectedReport.answers.map((answer, index) => (
                              <div key={`${answer.questionId || index}-${index}`} className="rounded-lg border border-gray-100 p-3 text-sm">
                                <p className="text-xs font-semibold uppercase text-gray-500">{answer.questionId || `Question ${index + 1}`}</p>
                                <p className="text-gray-900">Answer: <span className="font-medium">{answer.answer || '—'}</span></p>
                                {answer.description && <p className="text-xs text-gray-500">Note: {answer.description}</p>}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="mt-2 text-xs text-gray-500">No answers captured for this submission.</p>
                        )}
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-gray-900">Decision Notes</h4>
                        <p className="mt-2 text-sm text-gray-700">
                          {selectedReport.adminNotes || 'No admin notes were recorded for this report.'}
                        </p>
                        <p className="mt-1 text-xs text-gray-500">
                          Reviewed by {selectedReport.reviewedBy || 'N/A'} on{' '}
                          {selectedReport.reviewedAt?.toDate?.()?.toLocaleString?.() || 'Not captured'}
                        </p>
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-gray-900">AI Evidence Review</h4>
                        {imageAnalysisLoading && (
                          <p className="mt-2 text-xs text-blue-600">ChatGPT is analyzing the photo evidence…</p>
                        )}
                        {!imageAnalysisLoading && imageAnalysisError && (
                          <p className="mt-2 text-xs text-red-600">{imageAnalysisError}</p>
                        )}
                        {!imageAnalysisLoading && !imageAnalysisError && !selectedReport.imageValidation && (
                          <p className="mt-2 text-xs text-gray-500">
                            ChatGPT is preparing the image compliance summary. Please refresh if this message persists.
                          </p>
                        )}
                        {selectedReport.rejectionAnalysis && (
                          <div className="mt-3 space-y-2 rounded-lg border border-red-100 bg-red-50 p-3 text-sm text-red-900">
                            <p className="font-semibold">Rejection Reason: {selectedReport.rejectionAnalysis.reason}</p>
                            <p className="text-xs">{selectedReport.rejectionAnalysis.validationSummary}</p>
                            <p className="text-xs">
                              Model: {selectedReport.rejectionAnalysis.aiModel} · Generated{' '}
                              {new Date(selectedReport.rejectionAnalysis.generatedAt).toLocaleString()}
                            </p>
                          </div>
                        )}
                        {selectedReport.imageValidation && (
                          <div className="mt-3 space-y-3 rounded-lg border border-blue-100 bg-blue-50 p-3 text-sm text-blue-900">
                            <p>{selectedReport.imageValidation.overallFindings}</p>
                            {selectedReport.imageValidation.questionChecks?.length ? (
                              <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-blue-100 text-xs">
                                  <thead>
                                    <tr className="text-left text-blue-900">
                                      <th className="px-3 py-2 font-semibold">Question</th>
                                      <th className="px-3 py-2 font-semibold">Answer</th>
                                      <th className="px-3 py-2 font-semibold">AI Verdict</th>
                                      <th className="px-3 py-2 font-semibold">Summary</th>
                                    </tr>
                                  </thead>
                                  <tbody className="divide-y divide-blue-100">
                                    {selectedReport.imageValidation.questionChecks.map((check, index) => {
                                      const verdict = formatValidationStatus(check.status)
                                      return (
                                        <tr key={`validation-${check.questionId || index}`}>
                                          <td className="px-3 py-2">
                                            <p className="font-semibold">{check.questionLabel || getQuestionLabel(check.questionId)}</p>
                                            <p className="text-[11px] text-blue-800">{check.questionId}</p>
                                          </td>
                                          <td className="px-3 py-2">{check.answer || '—'}</td>
                                          <td className="px-3 py-2">
                                            <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${verdict.className}`}>
                                              {verdict.label}
                                            </span>
                                          </td>
                                          <td className="px-3 py-2">
                                            <p>{check.summary}</p>
                                            {check.photoEvidence?.length > 0 && (
                                              <div className="mt-1 flex flex-wrap gap-2">
                                                {check.photoEvidence.map((url, idx) => (
                                                  <a
                                                    key={`${check.questionId}-photo-${idx}`}
                                                    href={url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="text-[11px] underline"
                                                  >
                                                    Photo {idx + 1}
                                                  </a>
                                                ))}
                                              </div>
                                            )}
                                          </td>
                                        </tr>
                                      )
                                    })}
                                  </tbody>
                                </table>
                              </div>
                            ) : (
                              <p className="text-xs text-blue-900">
                                ChatGPT completed the scan but did not return per-question findings.
                              </p>
                            )}
                          </div>
                        )}
                        {!reportHasPhotos(selectedReport) && (
                          <p className="mt-2 text-xs text-gray-500">
                            No photo attachments were provided, so ChatGPT cannot match images to questions.
                          </p>
                        )}
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-gray-900">Photos & Attachments</h4>
                        <div className="mt-2 grid gap-3 md:grid-cols-3">
                          {aggregateReportPhotos(selectedReport).map(photo => (
                            <div key={photo} className="h-28 overflow-hidden rounded-lg border border-gray-100">
                              <a
                                href={photo}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="block h-full w-full transition-transform duration-200 hover:scale-105"
                                title="Open full-size photo"
                              >
                                <img src={photo} alt="Report evidence" className="h-full w-full object-cover" />
                              </a>
                            </div>
                          ))}
                          {aggregateReportPhotos(selectedReport).length === 0 && (
                            <div className="col-span-full rounded-lg border border-dashed border-gray-200 p-4 text-xs text-gray-500">
                              No photo evidence attached.
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </section>
                )}

              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function FeederPointFormModal({ title, feederPoint, onClose, onSave }: any) {
  const [formData, setFormData] = useState(feederPoint || {})

  const handleChange = (e: any) => {
    const { name, value } = e.target
    setFormData({ ...formData, [name]: value })
  }

  const handleSave = () => {
    onSave(formData)
  }

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75" onClick={onClose} />
        <div className="bg-white rounded-lg shadow-xl transform transition-all sm:max-w-lg sm:w-full">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900">{title}</h3>
            <div className="mt-4 space-y-4">
              <input type="text" name="name" value={formData.name || ''} onChange={handleChange} placeholder="Name" className="w-full px-3 py-2 border border-gray-300 rounded-md" />
              <input type="text" name="location.address" value={formData.location?.address || ''} onChange={handleChange} placeholder="Address" className="w-full px-3 py-2 border border-gray-300 rounded-md" />
              <select name="status" value={formData.status || ''} onChange={handleChange} className="w-full px-3 py-2 border border-gray-300 rounded-md">
                <option value="active">Active</option>
                <option value="maintenance">Maintenance</option>
                <option value="inactive">Inactive</option>
              </select>
              <select name="priority" value={formData.priority || ''} onChange={handleChange} className="w-full px-3 py-2 border border-gray-300 rounded-md">
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>
          <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
            <button type="button" onClick={handleSave} className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 sm:ml-3 sm:w-auto sm:text-sm">Save</button>
            <button type="button" onClick={onClose} className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm">Cancel</button>
          </div>
        </div>
      </div>
    </div>
  )
}

function DetailField({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-gray-100 p-3">
      <p className="text-xs font-semibold uppercase text-gray-500">{label}</p>
      <p className="text-sm text-gray-900">{value}</p>
    </div>
  )
}

function getStatusBadgeClasses(status: ComplianceReport['status'] | undefined) {
  switch (status) {
    case 'approved':
      return 'bg-green-100 text-green-800';
    case 'rejected':
      return 'bg-red-100 text-red-800';
    case 'requires_action':
      return 'bg-orange-100 text-orange-800';
    case 'pending':
    default:
      return 'bg-gray-100 text-gray-700';
  }
}

function formatValidationStatus(status: string | undefined) {
  switch (status) {
    case 'match':
      return { label: 'Match', className: 'bg-green-100 text-green-800' }
    case 'mismatch':
      return { label: 'Mismatch', className: 'bg-red-100 text-red-800' }
    case 'missing':
      return { label: 'Missing', className: 'bg-yellow-100 text-yellow-800' }
    default:
      return { label: 'Uncertain', className: 'bg-gray-100 text-gray-700' }
  }
}

function getQuestionLabel(questionId?: string) {
  if (!questionId) return 'Unknown Question'
  return QUESTION_LABEL_LOOKUP.get(questionId.toLowerCase()) || questionId
}

function buildQuestionLabelLookup(rules: ImageQuestionRule[]) {
  const map = new Map<string, string>()
  rules.forEach(rule => {
    rule.ids.forEach(id => {
      if (id) {
        map.set(id.toLowerCase(), rule.label)
      }
    })
  })
  return map
}

function reportHasPhotos(report: ComplianceReport | null) {
  if (!report) return false
  const attachmentHasUrl = report.attachments?.some(attachment => Boolean(attachment?.url))
  const answerHasPhoto = report.answers?.some(answer => answer.photos && answer.photos.length > 0)
  return Boolean(attachmentHasUrl || answerHasPhoto)
}

const QUESTION_KEY_MAP = {
  zone: ['Q1', 'q1', 'q1_zone_name', 'zone_name'],
  ward: ['Q2', 'q2', 'q2_ward_number', 'ward_number'],
  cleanliness: ['Q4', 'q4', 'q4_feeder_point_clean', 'feeder_point_clean', 'scp_area_clean'],
  segregation: ['Q7', 'q7', 'waste_segregated', 'wet_dry_waste_segregation'],
  vehicle: ['vehicle_separate_compartments', 'vehicle_available', 'vehicle_present'],
  swachWorkers: ['swach_workers_present', 'swach_workers_count', 'q10', 'Q10', 'q10_swach_workers_present'],
  surrounding: ['Q5', 'q5', 'q5_surrounding_area_clean', 'surrounding_area_clean', 'surrounding_area_maintained'],
  signboard: ['Q11', 'q11', 'q11_signboard_qr_display', 'signboard_qr_display', 'qr_display', 'visible_signboard'],
  overallScore: ['Q12', 'q12', 'overall_score', 'overall_compliance_rating']
}

const IMAGE_VALIDATION_RULES: ImageQuestionRule[] = [
  {
    label: 'Feeder Point Cleanliness',
    ids: QUESTION_KEY_MAP.cleanliness,
    expectation: 'Photos should show the feeder point ground area, swept and free from waste accumulations.'
  },
  {
    label: 'Waste Segregation',
    ids: QUESTION_KEY_MAP.segregation,
    expectation: 'Photos should show wet and dry waste stored separately or colour-coded bins in use.'
  },
  {
    label: 'Vehicle Availability',
    ids: QUESTION_KEY_MAP.vehicle,
    expectation: 'Photos should clearly show the assigned collection vehicle with registration visible.'
  },
  {
    label: 'Swach Worker Presence',
    ids: QUESTION_KEY_MAP.swachWorkers,
    expectation: 'Photos should capture uniformed workers or the deployed team at the site.'
  },
  {
    label: 'Signboard / QR Display',
    ids: QUESTION_KEY_MAP.signboard,
    expectation: 'Photos should showcase the feeder point signboard or QR panel clearly legible.'
  },
  {
    label: 'Overall Compliance Rating',
    ids: QUESTION_KEY_MAP.overallScore,
    expectation: 'Provide any supporting photos that justify the claimed overall rating.'
  }
]

const QUESTION_LABEL_LOOKUP = buildQuestionLabelLookup(IMAGE_VALIDATION_RULES)

function createFeederPointReportSummary(feederPoint: any, rawReports: ComplianceReport[] = []) {
  const reports = [...rawReports]
  reports.sort((a, b) => {
    const aTime = getReportTimestamp(a)?.getTime() || 0
    const bTime = getReportTimestamp(b)?.getTime() || 0
    return bTime - aTime
  })

  const latestReport = reports[0] || null
  const totalTripsSet = new Set<number>()
  reports.forEach(report => {
    if (report?.tripNumber) {
      totalTripsSet.add(report.tripNumber)
    }
  })

  const basicInfo = {
    name: feederPoint?.name || latestReport?.feederPointName || 'Not Available',
    zone: findAnswerAcrossReports(reports, QUESTION_KEY_MAP.zone) || feederPoint?.zone || 'Not Reported',
    ward: findAnswerAcrossReports(reports, QUESTION_KEY_MAP.ward) || feederPoint?.ward || 'Not Reported',
    date: latestReport ? formatDateDisplay(getReportTimestamp(latestReport)) : formatDateDisplay(new Date()),
    totalTrips: totalTripsSet.size > 0 ? String(totalTripsSet.size) : reports.length > 0 ? String(reports.length) : '0',
    submittedBy: latestReport?.submittedBy || latestReport?.userName || 'Not Available'
  }

  const cleanlinessFlag = interpretBoolean(findAnswerAcrossReports(reports, QUESTION_KEY_MAP.cleanliness))
  const segregationFlag = interpretBoolean(findAnswerAcrossReports(reports, QUESTION_KEY_MAP.segregation))
  const vehicleFlag = interpretBoolean(findAnswerAcrossReports(reports, QUESTION_KEY_MAP.vehicle))
  const areaFlag = interpretBoolean(findAnswerAcrossReports(reports, QUESTION_KEY_MAP.surrounding))
  const signboardFlag = interpretBoolean(findAnswerAcrossReports(reports, QUESTION_KEY_MAP.signboard))

  const overallStatus = {
    cleanliness: cleanlinessFlag === false ? 'Not Clean' : cleanlinessFlag === true ? 'Clean' : 'Data Pending',
    segregation: segregationFlag === false ? 'Not Segregated' : segregationFlag === true ? 'Segregated' : 'Data Pending',
    vehicle: vehicleFlag === false ? 'Not Present' : vehicleFlag === true ? 'Present' : 'Data Pending',
    swachWorkers: formatSwachWorkerPresence(findAnswerAcrossReports(reports, QUESTION_KEY_MAP.swachWorkers)),
    nearbyCleanliness: areaFlag === false ? 'No' : areaFlag === true ? 'Yes' : 'Data Pending',
    signboard: signboardFlag === false ? 'No' : signboardFlag === true ? 'Yes' : 'Data Pending',
    finalScore: formatFinalScore(findAnswerAcrossReports(reports, QUESTION_KEY_MAP.overallScore))
  }

  const tripSummaries = buildTripSummaries(reports)

  const observations = buildObservations(overallStatus, reports.length)

  const photoEvidence = tripSummaries.map((trip, index) => ({
    label: `Trip ${index + 1} Photo`,
    photoUrl: trip.photoUrl
  }))

  return {
    basicInfo,
    overallStatus,
    tripSummaries,
    observations,
    photoEvidence
  }
}

function buildTripSummaries(reports: ComplianceReport[]) {
  const tripOrder = [1, 2, 3]
  const tripMap = new Map<number, ComplianceReport>()

  reports.forEach(report => {
    if (report.tripNumber && !tripMap.has(report.tripNumber)) {
      tripMap.set(report.tripNumber, report)
    }
  })

  return tripOrder.map((tripNumber, index) => {
    const report = tripMap.get(tripNumber) || reports[index] || null
    const cleanAnswer = findAnswerInReport(report, QUESTION_KEY_MAP.cleanliness)
    const segregationAnswer = findAnswerInReport(report, QUESTION_KEY_MAP.segregation)
    const vehicleAnswer = findAnswerInReport(report, QUESTION_KEY_MAP.vehicle)
    const workersAnswer = findAnswerInReport(report, QUESTION_KEY_MAP.swachWorkers)

    const distance = typeof report?.distanceFromFeederPoint === 'number'
      ? formatDistance(report.distanceFromFeederPoint)
      : '—'

    const photoUrl = getPrimaryPhoto(report)

    return {
      tripLabel: `Trip ${tripNumber}`,
      time: report ? formatDateDisplay(getReportTimestamp(report), { hour: '2-digit', minute: '2-digit' }) : 'Pending',
      distance,
      clean: formatYesNo(interpretBoolean(cleanAnswer)),
      segregated: formatYesNo(interpretBoolean(segregationAnswer)),
      vehicle: formatYesNo(interpretBoolean(vehicleAnswer)),
      workers: formatSwachWorkerPresence(workersAnswer),
      photoAttached: photoUrl ? 'Yes' : 'No',
      photoUrl
    }
  })
}

function findAnswerAcrossReports(reports: ComplianceReport[], questionIds: string[]) {
  const lookup = questionIds.map(id => id.toLowerCase())
  for (const report of reports) {
    const answer = report?.answers?.find(entry => entry.questionId && lookup.includes(entry.questionId.toLowerCase()))
    if (answer) {
      return normalizeAnswer(answer.answer)
    }
  }
  return null
}

function findAnswerInReport(report: ComplianceReport | null, questionIds: string[]) {
  if (!report) return null
  const lookup = questionIds.map(id => id.toLowerCase())
  const entry = report.answers?.find(answer => answer.questionId && lookup.includes(answer.questionId.toLowerCase()))
  return entry ? normalizeAnswer(entry.answer) : null
}

function normalizeAnswer(answer: string | 'yes' | 'no' | undefined) {
  if (answer === undefined || answer === null) return null
  if (typeof answer === 'string') {
    return answer
  }
  return answer
}

function interpretBoolean(value: string | null) {
  if (!value) return null
  const normalized = value.trim().toLowerCase()
  if (['yes', 'y', 'true', 'clean', 'present', 'available', 'segregated', 'ok', 'good'].includes(normalized)) {
    return true
  }
  if (['no', 'n', 'false', 'not clean', 'absent', 'not present', 'not segregated', 'poor'].includes(normalized)) {
    return false
  }
  if (normalized === 'partial') {
    return false
  }
  return null
}

function formatYesNo(value: boolean | null) {
  if (value === true) return 'Yes'
  if (value === false) return 'No'
  return 'Data Pending'
}

function formatSwachWorkerPresence(value: string | null) {
  if (!value) return 'Data Pending'
  const numeric = parseInt(value, 10)
  if (!Number.isNaN(numeric)) {
    return numeric === 0 ? 'Not Present' : `${numeric} worker${numeric === 1 ? '' : 's'} present`
  }
  const booleanValue = interpretBoolean(value)
  if (booleanValue === true) return 'Present'
  if (booleanValue === false) return 'Not Present'
  return value
}

function formatFinalScore(value: string | null) {
  if (!value) return 'Average'
  const normalized = value.trim().toLowerCase()
  if (['excellent', 'good', 'average', 'poor'].includes(normalized)) {
    return normalized.charAt(0).toUpperCase() + normalized.slice(1)
  }
  const numeric = parseFloat(normalized)
  if (!Number.isNaN(numeric)) {
    if (numeric >= 4.5) return 'Excellent'
    if (numeric >= 3.5) return 'Good'
    if (numeric >= 2.5) return 'Average'
    return 'Poor'
  }
  return 'Average'
}

function formatDistance(distanceInMeters: number) {
  if (distanceInMeters >= 1000) {
    return `${(distanceInMeters / 1000).toFixed(1)} km`
  }
  return `${Math.round(distanceInMeters)} m`
}

function getReportTimestamp(report?: ComplianceReport | null) {
  if (!report) return null
  return (
    coerceDate(report.submittedAt) ||
    coerceDate(report.updatedAt) ||
    coerceDate(report.createdAt) ||
    (report.tripDate ? coerceDate(report.tripDate) : null)
  )
}

function coerceDate(value: any): Date | null {
  if (!value) return null
  if (value instanceof Date) return value
  if (typeof value === 'string') {
    const parsed = new Date(value)
    return Number.isNaN(parsed.getTime()) ? null : parsed
  }
  if (typeof value.toDate === 'function') {
    try {
      return value.toDate()
    } catch {
      return null
    }
  }
  return null
}

function formatDateDisplay(value: Date | null, options?: Intl.DateTimeFormatOptions) {
  if (!value) return 'Not Available'
  const formatterOptions = options || { year: 'numeric', month: 'short', day: 'numeric' }
  return value.toLocaleString(undefined, formatterOptions)
}

function getPrimaryPhoto(report: ComplianceReport | null) {
  if (!report) return null
  const attachment = report.attachments?.find(item => item.type === 'photo') || report.attachments?.[0]
  if (attachment?.url) {
    return attachment.url
  }
  for (const answer of report.answers || []) {
    if (answer.photos && answer.photos.length > 0) {
      return answer.photos[0]
    }
  }
  return null
}

function buildObservations(overallStatus: any, reportCount: number) {
  const observations: string[] = []

  if (reportCount === 0) {
    observations.push('No compliance submissions have been logged for this feeder point yet.')
  }
  observations.push(`Feeder point cleanliness is noted as ${overallStatus.cleanliness.toLowerCase()}.`)
  observations.push(`Segregation check outcome is ${overallStatus.segregation.toLowerCase()}.`)
  observations.push(`Swach worker presence recorded as ${overallStatus.swachWorkers}.`)
  observations.push(`Signboard / QR visibility marked ${overallStatus.signboard}.`)

  return observations.slice(0, 5)
}

function formatReportDate(report: ComplianceReport) {
  const timestamp = getReportTimestamp(report)
  return timestamp ? timestamp.toLocaleString() : null
}

function aggregateReportPhotos(report: ComplianceReport | null) {
  if (!report) return []
  const photos: string[] = []
  report.attachments?.forEach(attachment => {
    if (attachment.url) {
      photos.push(attachment.url)
    }
  })
  report.answers?.forEach(answer => {
    if (answer.photos) {
      answer.photos.forEach(url => {
        if (url) photos.push(url)
      })
    }
  })
  return Array.from(new Set(photos))
}
