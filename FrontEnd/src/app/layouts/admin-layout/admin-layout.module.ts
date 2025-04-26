import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { AdminLayoutRoutes } from './admin-layout.routing';

import { HomeComponent } from '../../home/home.component';
import { InspectionComponent } from '../../inspection/inspection.component';
import { DamageStatsComponent } from '../../damage-stats/damage-stats.component';
import { ReportsComponent } from '../../reports/reports.component';
import { SettingsComponent } from '../../settings/settings.component';

import { LbdModule } from '../../lbd/lbd.module'; // ✅ lbd-module 추가

@NgModule({
  imports: [
    CommonModule,
    RouterModule.forChild(AdminLayoutRoutes),
    FormsModule,
    LbdModule // ✅ 반드시 추가
  ],
  declarations: [
    HomeComponent,
    InspectionComponent,
    DamageStatsComponent,
    ReportsComponent,
    SettingsComponent
  ]
})
export class AdminLayoutModule {}
