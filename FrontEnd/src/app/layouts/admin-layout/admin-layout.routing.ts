import { Routes } from '@angular/router';

import { HomeComponent } from '../../home/home.component';
import { InspectionComponent } from '../../inspection/inspection.component';
import { DamageStatsComponent } from '../../damage-stats/damage-stats.component';
import { MappingComponent } from '../../mapping/mapping.component';
import { AnalyzeComponent } from '../../analyze/analyze.component';
import { InfoComponent } from '../../info/info.component';

export const AdminLayoutRoutes: Routes = [
    { path: 'dashboard', component: HomeComponent },
    { path: 'inspection', component: InspectionComponent },
    { path: 'damage-stats', component: DamageStatsComponent },
    { path: 'mapping', component: MappingComponent },     // ✅ 추가
    { path: 'analyze', component: AnalyzeComponent },      // ✅ 추가
    { path: 'info', component: InfoComponent },          // ✅ 추가
];
