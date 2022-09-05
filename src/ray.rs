use crate::rect::Rect;

pub struct Ray {
    pub pos: cgmath::Vector2<f32>,
    pub dir: cgmath::Vector2<f32>,
}

impl Ray {
    pub fn intersects(&self, rect: &Rect) -> bool {
        self.intersection_distance(rect).is_some()
    }

    pub fn intersection_point(&self, rect: &Rect) -> Option<cgmath::Vector2<f32>> {
        let dist = self.intersection_distance(rect);

        if let Some(d) = dist {
            Some(d * self.dir + self.pos)
        } else {
            None
        }
    }

    // A ray will intersect a rect unless either:
    // - The ray passes the rect on the y axis before reaching it on the x axis,
    // - The ray passes the rect on the x axis before reaching it on the y axis,
    pub fn intersection_distance(&self, rect: &Rect) -> Option<f32> {
        let mut t_min = -f32::INFINITY;
        let mut t_max = f32::INFINITY;

        if self.dir.x != 0.0 {
            let tx1 = (rect.min.x - self.pos.x) / self.dir.x;
            let tx2 = (rect.max.x - self.pos.x) / self.dir.x;

            t_min = t_min.max(tx1.min(tx2));
            t_max = t_max.min(tx1.max(tx2));
        }

        if self.dir.y != 0.0 {
            let ty1 = (rect.min.y - self.pos.y) / self.dir.y;
            let ty2 = (rect.max.y - self.pos.y) / self.dir.y;

            t_min = t_min.max(ty1.min(ty2));
            t_max = t_max.min(ty1.max(ty2));
        }

        let hit = t_max >= 0.0 && t_max >= t_min;

        if hit {
            Some(t_min)
        } else {
            None
        }
    }
}
