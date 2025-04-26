import { Routes } from '@angular/router';

import { HomeComponent } from '../../home/home.component';
import { InspectionComponent } from '../../inspection/inspection.component';
import { DamageStatsComponent } from '../../damage-stats/damage-stats.component';
import { ReportsComponent } from '../../reports/reports.component';
import { SettingsComponent } from '../../settings/settings.component';

export const AdminLayoutRoutes: Routes = [
    { path: 'dashboard', component: HomeComponent },
    { path: 'inspection', component: InspectionComponent },
    { path: 'damage-stats', component: DamageStatsComponent },
    { path: 'reports', component: ReportsComponent },
    { path: 'settings', component: SettingsComponent },
];
