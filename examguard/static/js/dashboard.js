import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, Users, AlertTriangle, Eye } from 'lucide-react';

const Dashboard = () => {
  const [stats, setStats] = useState({
    peopleCount: 0,
    headTurns: 0,
    handsUp: 0,
    warnings: 0
  });

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Exam Monitoring Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">People Detected</p>
                <p className="text-2xl font-bold">{stats.peopleCount}</p>
              </div>
              <Users className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Head Turns</p>
                <p className="text-2xl font-bold">{stats.headTurns}</p>
              </div>
              <Eye className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Hands Up</p>
                <p className="text-2xl font-bold">{stats.handsUp}</p>
              </div>
              <Activity className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Active Warnings</p>
                <p className="text-2xl font-bold">{stats.warnings}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Live Feed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
              <img 
                src="http://localhost:5001/video_feed" 
                alt="Live video feed"
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.target.src = `http://localhost:5001/video_feed?${new Date().getTime()}`;
                }}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Events</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Example events - you can populate this with real data */}
              <div className="flex items-center space-x-4 p-3 bg-gray-50 rounded-lg">
                <AlertTriangle className="h-5 w-5 text-yellow-500" />
                <div>
                  <p className="text-sm font-medium">Head Turn Detected</p>
                  <p className="text-xs text-gray-500">Person 1 - 30° right</p>
                </div>
              </div>
              <div className="flex items-center space-x-4 p-3 bg-gray-50 rounded-lg">
                <Activity className="h-5 w-5 text-blue-500" />
                <div>
                  <p className="text-sm font-medium">Hand Raised</p>
                  <p className="text-xs text-gray-500">Person 2 - Right hand</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;